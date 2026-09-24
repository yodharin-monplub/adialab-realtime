"""Outer-holdout evaluation of Submission31 (reuses Model123's chain/Sub30 rows).

Runs the sealed-candidate bundle's SeriesPredictor on the 1,667 holdout
series (8 workers), then joins with models/model123/holdout_predictions.npz
(same series, same order) to report Sub28 chain / Sub30 / Sub31 TS-AUC.

Usage: python models/model124/holdout_eval31.py [--workers 8]
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
SUB = os.environ.get('SUBMISSION_NAME', 'submission31')
BUNDLE = ROOT / 'submissions' / SUB
HERE = Path(__file__).resolve().parent
SUFFIX = '31' if SUB == 'submission31' else SUB.replace('submission', '')
sys.path.insert(0, str(BUNDLE))
sys.path.insert(1, str(ROOT))

_bundle = None
_frame = None


def init_worker():
    global _bundle, _frame
    import solution124
    _bundle = solution124.Bundle(BUNDLE / 'resources')
    _frame = pd.read_parquet(ROOT / 'train/X_train.parquet')


def run_series(sid):
    import numpy as np
    import solution124
    series = _frame.loc[sid]
    history = series.loc[series.period == 1, 'value'].to_numpy(dtype=np.float32)
    online = series.loc[series.period == 2, 'value'].to_numpy(dtype=np.float32)
    predictor = solution124.SeriesPredictor(_bundle, history)
    return int(sid), np.asarray([predictor.step(v) for v in online], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    ref = np.load(ROOT / 'models/model123/holdout_predictions.npz')
    ids = ref['ids']
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
    sub31 = np.concatenate([final[int(s)] for s in ids])
    assert len(sub31) == len(ref['target'])
    from tsauc import tsauc
    y, t = ref['target'], ref['step']
    result = {'bundle': SUB, 'series': int(len(ids)), 'rows': int(len(y)),
              'submission28_chain': tsauc(y, ref['chain'], t),
              'submission30_final': tsauc(y, ref['final'], t),
              f'{SUB}_final': tsauc(y, sub31, t),
              'seconds': time.perf_counter() - started}
    result['gain_vs_sub30'] = result[f'{SUB}_final'] - result['submission30_final']
    if SUB != 'submission31' and (HERE / 'holdout_predictions31.npz').exists():
        s31 = np.load(HERE / 'holdout_predictions31.npz')['final']
        result['submission31_final'] = tsauc(y, s31, t)
        result['gain_vs_sub31'] = result[f'{SUB}_final'] - result['submission31_final']
    np.savez(HERE / f'holdout_predictions{SUFFIX}.npz', final=sub31, ids=ids)
    (HERE / f'holdout_results{SUFFIX}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
