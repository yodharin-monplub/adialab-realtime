"""Validate Submission31: GRU streaming parity, chain parity, determinism, inference-only.

  python models/model124/deployment/validate31.py parity
  python models/model124/deployment/validate31.py benchmark --series 24 --workers 2
Offline reference for the GRU component: numpy GRUs run on the exported
training inputs (data/sequences.npz) for the same development series.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[3]
SUB = os.environ.get('SUBMISSION_NAME', 'submission31')
BUNDLE = ROOT / 'submissions' / SUB
OUT = ROOT / 'models/model124/deployment_validation' / ('' if SUB == 'submission31' else SUB)
sys.path.insert(0, str(BUNDLE))
sys.path.insert(1, str(ROOT))


def write(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def records(count, seed=16312):
    with np.load(ROOT / 'models/model5/artifacts/split_assignments.npz') as p:
        positions = np.flatnonzero(p['inner_series_fold'] >= 0)
    with np.load(ROOT / 'models/artifacts/core_features.metadata.npz') as m:
        chosen = np.random.default_rng(seed).choice(positions, count, replace=False)
        ids = m['series_ids'][chosen]
    frame = ds.dataset(ROOT / 'train/X_train.parquet', format='parquet').to_table(
        filter=ds.field('id').isin(ids.tolist())).to_pandas().sort_index()
    out = []
    for position, sid in zip(chosen, ids):
        series = frame.loc[sid]
        out.append((int(sid), int(position),
                    series.loc[series.period == 1, 'value'].to_numpy(dtype=np.float32),
                    series.loc[series.period == 2, 'value'].to_numpy(dtype=np.float32)))
    return out


def parity():
    from solution124 import Bundle, SeriesPredictor
    from gru_numpy import NumpyGRU
    from models.model10.sources import load_development_layout
    layout = load_development_layout(ROOT)
    bundle = Bundle(BUNDLE / 'resources')
    spec = bundle.manifest['model124_gru']
    data_files = ['models/model124/data/sequences.npz', 'models/model124/data/sequences_d0.npz']
    Xs = [np.load(ROOT / p)['X'] for p in data_files[:len(spec.get('input_contracts') or [1])]]
    sample = records(3)
    grus = [(NumpyGRU(dict(np.load(BUNDLE / 'resources' / f['path'])), int(f['layers'])), int(f.get('contract', 0)))
            for f in spec['folds']]
    worst_gru, worst_final, rows = 0.0, 0.0, 0
    predictions = []
    for _, position, history, online in sample:
        n = min(180, len(online))
        idx = np.flatnonzero(layout.series_position == position)[:n]
        assert np.array_equal(layout.time_online[idx], np.arange(n))
        xs_off = [X[idx].astype(np.float32) for X in Xs]
        expected_gru = np.mean([g.run(xs_off[c]) for g, c in grus], axis=0)
        predictor = SeriesPredictor(bundle, history)
        # capture the streaming GRU logit by re-deriving it from the score components
        streamed = []
        for i, v in enumerate(online[:n]):
            p = predictor.step(v)
            predictions.append(p)
            streamed.append(p)
        # invert the final score to the GRU term is not possible alone; instead recompute
        # the streamed GRU logits by feeding the streaming inputs through fresh GRUs
        pred2 = SeriesPredictor(bundle, history)
        states = [g.initial_state() for g, _ in grus]
        logits = []
        for i, v in enumerate(online[:n]):
            row = np.asarray(pred2.detector.step(float(v)), dtype=np.float32)
            np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
            np.clip(row, -1e6, 1e6, out=row)
            white = pred2.white.step(float(v))
            xi = bundle.gru_inputs(row, white, i)
            logits.append(float(np.mean([g.step(xi[c], s) for (g, c), s in zip(grus, states)], dtype=np.float32)))
        worst_gru = max(worst_gru, float(np.max(np.abs(np.asarray(logits) - expected_gru))))
        rows += n
    second = []
    for _, _, history, online in sample:
        predictor = SeriesPredictor(bundle, history)
        second.extend(predictor.step(v) for v in online[:min(180, len(online))])
    np.testing.assert_array_equal(predictions, second)
    from main import infer, train
    touched = []

    def stream():
        for i, v in enumerate(sample[0][3][:10]):
            touched.append(i)
            yield v
    output = infer([(sample[0][2], stream())], str(BUNDLE / 'resources'))
    assert next(output) is None and not touched
    for i in range(10):
        assert math.isfinite(next(output)) and len(touched) == i + 1

    class NoData:
        def __iter__(self):
            raise AssertionError('Training data consumed')
    train(NoData(), str(BUNDLE / 'resources'))
    result = {'rows': rows, 'gru_streaming_vs_offline_max_abs_logit_diff': worst_gru,
              'deterministic_replay': True, 'incremental_consumption': True,
              'train_does_not_consume_data': True, 'python': sys.version}
    if worst_gru > 1e-3:
        raise AssertionError(f'GRU streaming parity failed: {worst_gru}')
    write('parity.json', result)
    print(json.dumps(result, indent=2))


_bundle = None


def init_worker(module):
    global _bundle
    sys.path.insert(0, str(BUNDLE))
    mod = __import__(module)
    _bundle = (mod, mod.Bundle(BUNDLE / 'resources'))


def timed(record):
    mod, bundle = _bundle
    sid, _, h, o = record
    started = time.perf_counter()
    p = mod.SeriesPredictor(bundle, h)
    values = np.asarray([p.step(v) for v in o], dtype=np.float32)
    return {'id': sid, 'rows': len(o), 'seconds': time.perf_counter() - started,
            'checksum': float(values.astype(float).sum())}


def benchmark(count, workers):
    sample = records(count)
    report = {}
    for module in ('solution112', 'solution124'):
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(module,)) as pool:
            results = list(pool.map(timed, sample))
        report[module] = {'workers': workers, 'series': count, 'rows': sum(x['rows'] for x in results),
                          'wall_seconds': time.perf_counter() - started,
                          'worker_seconds': sum(x['seconds'] for x in results)}
    report['ratio_worker_seconds'] = report['solution124']['worker_seconds'] / report['solution112']['worker_seconds']
    report['estimated_cloud_hours'] = 3.93 * report['ratio_worker_seconds']  # Submission30 actual: 14,153 s
    write(f'benchmark_w{workers}_n{count}.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('parity', 'benchmark'))
    parser.add_argument('--series', type=int, default=24)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    parity() if args.command == 'parity' else benchmark(args.series, args.workers)
