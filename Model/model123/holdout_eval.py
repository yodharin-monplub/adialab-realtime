"""Model123: outer-holdout evaluation of Submission28 vs Submission30.

The 1,667 series with inner_series_fold == -1 were sealed by Model5 and never
used for any fitting or selection (teacher features exist only for the 8,333
development series).  The sealed Submission30 bundle is run on them exactly as
the cloud would run it; each online step yields both the Submission28 chain
score and the Submission30 blended score.  TS-AUC of both is reported.

Usage: python models/model123/holdout_eval.py [--workers 8] [--limit N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUB = os.environ.get('SUBMISSION_NAME', 'submission30')  # e.g. submission31
BUNDLE = ROOT / 'submissions' / SUB
HERE = Path(__file__).resolve().parent
SUFFIX = '' if SUB == 'submission30' else f'_{SUB}'
sys.path.insert(0, str(BUNDLE))
sys.path.insert(1, str(ROOT))


def holdout_series():
    with np.load(ROOT / 'models/model5/artifacts/split_assignments.npz') as p:
        positions = np.flatnonzero(p['inner_series_fold'] < 0)
    with np.load(ROOT / 'models/artifacts/core_features.metadata.npz') as m:
        ids = m['series_ids'][positions]
    tau = pd.read_parquet(ROOT / 'train/y_train_index.parquet')['tau_index']
    return ids, tau


_bundle = None
_frame = None


def init_worker():
    global _bundle, _frame
    import solution112
    _bundle = solution112.Bundle(BUNDLE / 'resources')
    _frame = pd.read_parquet(ROOT / 'train/X_train.parquet')


def run_series(sid):
    import math
    import numpy as np
    from solution26 import logistic, logit
    from primitive_runtime import _bag_probabilities
    import solution112
    series = _frame.loc[sid]
    history = series.loc[series.period == 1, 'value'].to_numpy(dtype=np.float32)
    online = series.loc[series.period == 2, 'value'].to_numpy(dtype=np.float32)
    predictor = solution112.SeriesPredictor(_bundle, history)
    bundle = _bundle
    chain_out, final_out = [], []
    for value in online:
        # Mirror of solution112.SeriesPredictor.step, additionally exposing the chain score.
        value = float(value)
        row = np.asarray(predictor.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        white = predictor.white.step(value)
        probability = bundle.primitive_probabilities(row, white)
        primitive_margin = logit(probability).astype(np.float32)
        margin9 = np.float32(primitive_margin.astype(np.float64) @ bundle.coefficient)
        p9 = np.float32(logistic(float(margin9)))
        current = float(logit(float(p9), epsilon=1e-7))
        index = predictor.step_index
        predictor.step_index += 1
        while predictor.peaks and predictor.peaks[0][0] <= index - 128:
            predictor.peaks.popleft()
        while predictor.peaks and predictor.peaks[-1][1] <= current:
            predictor.peaks.pop()
        predictor.peaks.append((index, current))
        transformed = np.float32(current + .1 * (predictor.peaks[0][1] - current))
        compact_row = predictor.compact.update(value)[None, :]
        p11 = np.float32(_bag_probabilities(
            [model.predict(compact_row, num_threads=1)[0] for model in bundle.compact_models]))
        margin11 = np.float32(logit(p11))
        combined = float(np.float32(.9 * transformed + .1 * margin11))
        pd_ = float(bundle.distilled_probability(row, white))
        pd_ = min(max(pd_, 1e-6), 1 - 1e-6)
        distilled = math.log(pd_) - math.log1p(-pd_)
        score = ((combined - bundle.mu_chain) / bundle.sd_chain
                 + bundle.weight * (distilled - bundle.mu_dist) / bundle.sd_dist)
        chain_out.append(combined)
        final_out.append(float(logistic(score)))
    return int(sid), np.asarray(chain_out, np.float32), np.asarray(final_out, np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    ids, tau = holdout_series()
    if args.limit:
        ids = ids[:args.limit]
    started = time.perf_counter()
    chain, final, target, step = [], [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=init_worker) as pool:
        for sid, c, f in pool.map(run_series, ids.tolist(), chunksize=4):
            n = len(c)
            ti = int(tau.loc[sid])
            y = np.zeros(n, np.int8) if ti < 0 else (np.arange(n) >= ti).astype(np.int8)
            chain.append(c); final.append(f); target.append(y); step.append(np.arange(n, dtype=np.int32))
            done += 1
            if done % 100 == 0:
                print(f'{done}/{len(ids)} series, {time.perf_counter() - started:.0f}s', flush=True)
    from tsauc import tsauc
    chain, final = np.concatenate(chain), np.concatenate(final)
    target, step = np.concatenate(target), np.concatenate(step)
    np.savez(HERE / f'holdout_predictions{SUFFIX}.npz', chain=chain, final=final, target=target, step=step, ids=ids)
    result = {'bundle': SUB, 'series': int(len(ids)), 'rows': int(len(target)),
              'submission28_chain': tsauc(target, chain, step),
              f'{SUB}_final': tsauc(target, final, step),
              'seconds': time.perf_counter() - started}
    result['gain'] = result[f'{SUB}_final'] - result['submission28_chain']
    (HERE / f'holdout_results{SUFFIX}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
