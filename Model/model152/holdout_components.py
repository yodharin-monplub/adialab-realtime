"""One streaming pass over the 1,667 sealed holdout series that SAVES the blend components, so every later
student recipe can be scored on the holdout offline (minutes instead of ~2.5 h per candidate).

Per online row (order = models/model123/holdout_predictions.npz):
  chain    the Submission28 chain margin ("combined" in holdout_eval147.py)
  gru      mean GRU logit of the 15 Submission31 GRUs
  X        527 float32 student inputs: 439 detector union columns (protocol union_names order, NaN->0,
           clipped), 61 whitening columns (model73 FEATURE_NAMES order), 27 whitened split tests (NAMES order)
Resumable: writes into memory-mapped .npy files, done-flags per series.
Output: models/model152/holdout/{X.npy, chain.npy, gru.npy, done.npy}
Usage: python models/model152/holdout_components.py [--workers 6] [--limit N]
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

os.environ['SP_WHITEN'] = '1'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'submissions/submission31'
OUT = Path(__file__).resolve().parent / 'holdout'
sys.path.insert(0, str(BUNDLE))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / 'models/model144'))
PROTOCOL = ROOT / 'models/model84/runs/e1e98a6e8af2df25/protocol.json'

_bundle = _frame = _det = None


def det_union_names():
    union = json.loads(PROTOCOL.read_text())['contract']['union_names']
    return [n for n in union if not n.startswith('white_')]


def init_worker():
    global _bundle, _frame, _det
    import solution124
    _bundle = solution124.Bundle(BUNDLE / 'resources')
    _frame = pd.read_parquet(ROOT / 'train/X_train.parquet')
    by_name = {name: i for i, name in enumerate(_bundle.manifest['feature_names'])}
    _det = np.asarray([by_name[n] for n in det_union_names()], np.int64)


def run_series(sid):
    import numpy as np
    from solution26 import logistic, logit
    from primitive_runtime import _bag_probabilities
    import solution124
    from split_tests import StreamingSplitTests as FastTests
    series = _frame.loc[sid]
    history = series.loc[series.period == 1, 'value'].to_numpy(dtype=np.float32)
    online = series.loc[series.period == 2, 'value'].to_numpy(dtype=np.float32)
    predictor = solution124.SeriesPredictor(_bundle, history)
    tests = FastTests(history)
    bundle = _bundle
    X = np.empty((len(online), len(_det) + 61 + 27), np.float32)
    chain = np.empty(len(online), np.float32); gru = np.empty(len(online), np.float32)
    for i, value in enumerate(online):
        value = float(value)
        row = np.asarray(predictor.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        white = np.asarray(predictor.white.step(value), dtype=np.float32)
        ft = np.asarray(tests.step(value), dtype=np.float32)
        np.nan_to_num(ft, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        probability = bundle.primitive_probabilities(row, white)
        margin9 = np.float32(logit(probability).astype(np.float32).astype(np.float64) @ bundle.coefficient)
        current = float(logit(float(np.float32(logistic(float(margin9)))), epsilon=1e-7))
        index = predictor.step_index
        predictor.step_index += 1
        while predictor.peaks and predictor.peaks[0][0] <= index - 128:
            predictor.peaks.popleft()
        while predictor.peaks and predictor.peaks[-1][1] <= current:
            predictor.peaks.pop()
        predictor.peaks.append((index, current))
        transformed = np.float32(current + .1 * (predictor.peaks[0][1] - current))
        compact_row = predictor.compact.update(value)[None, :]
        p11 = np.float32(_bag_probabilities([m.predict(compact_row, num_threads=1)[0] for m in bundle.compact_models]))
        chain[i] = np.float32(.9 * transformed + .1 * np.float32(logit(p11)))
        x = bundle.gru_input(row, white, index)
        gru[i] = np.mean([g.step(x, s) for g, s in zip(bundle.grus, predictor.gru_states)], dtype=np.float32)
        X[i, :len(_det)] = row[_det]; X[i, len(_det):len(_det) + 61] = white; X[i, len(_det) + 61:] = ft
    return int(sid), X, chain, gru


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    ref = np.load(ROOT / 'models/model123/holdout_predictions.npz')
    ids, step = ref['ids'], ref['step']
    starts = np.flatnonzero(step == 0); assert len(starts) == len(ids)
    ends = np.r_[starts[1:], len(step)]
    where = {int(s): (a_, b_) for s, a_, b_ in zip(ids, starts, ends)}
    OUT.mkdir(parents=True, exist_ok=True)
    width = len(det_union_names()) + 61 + 27
    mode = lambda p: 'r+' if p.exists() else 'w+'  # noqa: E731
    X = np.lib.format.open_memmap(OUT / 'X.npy', mode=mode(OUT / 'X.npy'), dtype=np.float32, shape=(len(step), width))
    chain = np.lib.format.open_memmap(OUT / 'chain.npy', mode=mode(OUT / 'chain.npy'), dtype=np.float32, shape=(len(step),))
    gru = np.lib.format.open_memmap(OUT / 'gru.npy', mode=mode(OUT / 'gru.npy'), dtype=np.float32, shape=(len(step),))
    done_path = OUT / 'done.npy'
    done = set(np.load(done_path).tolist()) if done_path.exists() else set()
    todo = [int(s) for s in (ids[:a.limit] if a.limit else ids) if int(s) not in done]
    print(f'{len(done)} done, {len(todo)} to go, width {width}', flush=True)
    t0 = time.perf_counter(); n = 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        for sid, x, c, g in pool.map(run_series, todo, chunksize=4):
            lo, hi = where[sid]; assert hi - lo == len(c)
            X[lo:hi] = x; chain[lo:hi] = c; gru[lo:hi] = g
            done.add(sid); n += 1
            if n % 50 == 0 or n == len(todo):
                X.flush(); chain.flush(); gru.flush(); np.save(done_path, np.asarray(sorted(done), np.int64))
                print(f'{len(done)}/{len(ids)} series, {time.perf_counter() - t0:.0f}s', flush=True)
    print('finished', flush=True)


if __name__ == '__main__':
    main()
