"""Outer-holdout evaluation of the stage-4 candidate: Submission31 with the distilled voters
replaced by the Model140 arm-E voters (317 + 13 test features, correlation-pruned).

score = z(chain) + w * z(logit E-bag) + w_gru * z(gru), z-constants of E from
models/model112/constants_e140.json (select_blend.py), everything else from the sealed
Submission31 bundle.  Compares with Submission31's own holdout predictions.
Usage: python models/model140/holdout_eval140.py [--workers 8] [--arm E]
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
BUNDLE = ROOT / 'submissions/submission31'
HERE = Path(__file__).resolve().parent
ARM = os.environ.get('ARM140', 'E')
sys.path.insert(0, str(BUNDLE))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(HERE))

_bundle = None
_frame = None
_voters = None
_const = None


def load_voters(bundle):
    import lightgbm as lgb
    from model73_whitening import FEATURE_NAMES
    from fast_online import NAMES
    protocol = json.loads((ROOT / 'models/model84/runs/e1e98a6e8af2df25/protocol.json').read_text())
    by_name = {name: number for number, name in enumerate(bundle.manifest['feature_names'])}
    white_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    ft_index = {name: i for i, name in enumerate(NAMES)}
    voters = []
    for fold in range(5):
        names = list(protocol['contract']['names'][fold]) + list(NAMES)
        cols = np.load(HERE / f'runs/{ARM}_fold{fold}.npz')['columns']
        chosen = [names[c] for c in cols]
        det = [by_name[n] for n in chosen if n in by_name and n not in white_index and n not in ft_index]
        white = [white_index[n] for n in chosen if n in white_index]
        ft = [ft_index[n] for n in chosen if n in ft_index]
        assert len(det) + len(white) + len(ft) == len(chosen)
        # training column order was ascending index: detector block, whitening block, test block
        booster = lgb.Booster(model_file=str(HERE / f'runs/{ARM}_fold{fold}.txt'))
        assert booster.num_feature() == len(chosen)
        voters.append((booster, np.asarray(det, np.int32), np.asarray(white, np.int32), np.asarray(ft, np.int32)))
    return voters


def init_worker():
    global _bundle, _frame, _voters, _const
    import solution124
    _bundle = solution124.Bundle(BUNDLE / 'resources')
    _frame = pd.read_parquet(ROOT / 'train/X_train.parquet')
    _voters = load_voters(_bundle)
    _const = json.loads((ROOT / 'models/model112/constants_e140.json').read_text())


def run_series(sid):
    import math
    import numpy as np
    from solution26 import logistic, logit
    from primitive_runtime import _bag_probabilities
    import solution124
    from fast_online import FastTests
    series = _frame.loc[sid]
    history = series.loc[series.period == 1, 'value'].to_numpy(dtype=np.float32)
    online = series.loc[series.period == 2, 'value'].to_numpy(dtype=np.float32)
    predictor = solution124.SeriesPredictor(_bundle, history)
    tests = FastTests(history)
    bundle = _bundle
    out = []
    for value in online:
        value = float(value)
        row = np.asarray(predictor.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        white = np.asarray(predictor.white.step(value), dtype=np.float32)
        ft = tests.step(value)
        np.nan_to_num(ft, copy=False, nan=0., posinf=1e6, neginf=-1e6)
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
        probs = []
        for booster, det, wh, fi in _voters:
            x = np.concatenate((row[det], white[wh], ft[fi])).astype(np.float32)[None, :]
            probs.append(np.float32(booster.predict(x, num_threads=1)[0]))
        pe = float(_bag_probabilities(probs))
        pe = min(max(pe, 1e-6), 1 - 1e-6)
        dist = math.log(pe) - math.log1p(-pe)
        x = bundle.gru_input(row, white, index)
        gru_logit = float(np.mean([gru.step(x, state) for gru, state in zip(bundle.grus, predictor.gru_states)],
                                  dtype=np.float32))
        score = ((combined - bundle.mu_chain) / bundle.sd_chain
                 + bundle.weight * (dist - _const['mu_dist']) / _const['sd_dist']
                 + bundle.weight_gru * (gru_logit - bundle.mu_gru) / bundle.sd_gru)
        out.append(float(logistic(score)))
    return int(sid), np.asarray(out, np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    ref = np.load(ROOT / 'models/model123/holdout_predictions.npz')
    ids = ref['ids']
    if args.limit:
        ids = ids[:args.limit]
    started = time.perf_counter()
    final = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=init_worker) as pool:
        for sid, f in pool.map(run_series, ids.tolist(), chunksize=4):
            final[sid] = f
            done += 1
            if done % 100 == 0:
                print(f'{done}/{len(ids)} series, {time.perf_counter() - started:.0f}s', flush=True)
    cand = np.concatenate([final[int(s)] for s in ids])
    from tsauc import tsauc
    n = len(cand)
    y, t = ref['target'][:n], ref['step'][:n]
    s31 = np.load(ROOT / 'models/model124/holdout_predictions31.npz')['final'][:n]
    result = {'candidate': f'submission31 + Model140 arm {ARM} voters', 'series': int(len(ids)), 'rows': int(n),
              'submission28_chain': tsauc(y, ref['chain'][:n], t),
              'submission30_final': tsauc(y, ref['final'][:n], t),
              'submission31_final': tsauc(y, s31, t),
              'candidate_final': tsauc(y, cand, t),
              'seconds': time.perf_counter() - started}
    result['gain_vs_sub31'] = result['candidate_final'] - result['submission31_final']
    np.savez(HERE / f'holdout_predictions_{ARM}.npz', final=cand, ids=ids)
    (HERE / f'holdout_results_{ARM}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
