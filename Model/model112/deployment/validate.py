"""Validate Submission30: streaming parity, offline-cache parity, inference-only, benchmark.

  python models/model112/deployment/validate.py parity
  python models/model112/deployment/validate.py benchmark --series 24 --workers 2
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
SUB = os.environ.get('SUBMISSION_NAME', 'submission30')  # e.g. submission31
BUNDLE = ROOT / 'submissions' / SUB
BASE = ROOT / 'submissions/submission28'
OUT = ROOT / 'models/model112/deployment_validation' / ('' if SUB == 'submission30' else SUB)
sys.path.insert(0, str(BUNDLE))
sys.path.insert(1, str(ROOT))


def write(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def records(count):
    with np.load(ROOT / 'models/model5/artifacts/split_assignments.npz') as p:
        positions = np.flatnonzero(p['inner_series_fold'] >= 0)
    with np.load(ROOT / 'models/artifacts/core_features.metadata.npz') as m:
        chosen = np.random.default_rng(16312).choice(positions, count, replace=False)
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
    from solution112 import Bundle, SeriesPredictor
    import solution73
    from models.model10.sources import load_development_layout
    layout = load_development_layout(ROOT)
    bundle = Bundle(BUNDLE / 'resources')
    spec = bundle.manifest['model112_distilled']
    golden_sample = records(3)
    chain_predictions, predictions, offline = [], [], []
    real = np.load(ROOT / 'models/model20/runs/82abe027caea808d/real_features.npy', mmap_mode='r')
    white = np.load(ROOT / 'models/model57/runs/dfab0ede1c3206a4/real_white.npy', mmap_mode='r')
    union = json.loads((ROOT / 'models/model20/runs/82abe027caea808d/real_features.cache.json').read_text())['contract']['names']
    import lightgbm as lgb
    boosters = [lgb.Booster(model_file=str(BUNDLE / 'resources' / v['path'])) for v in spec['voters']]
    base_bundle = solution73.Bundle(BUNDLE / 'resources')
    for _, position, history, online in golden_sample:
        n = min(180, len(online))
        base_predictor = solution73.SeriesPredictor(base_bundle, history)
        chain_predictions.extend(base_predictor.step(v) for v in online[:n])
        predictor = SeriesPredictor(bundle, history)
        predictions.extend(predictor.step(v) for v in online[:n])
        rows = np.flatnonzero(layout.series_position == position)[:n]
        assert np.array_equal(layout.time_online[rows], np.arange(n))
        feats = np.hstack([np.asarray(real[rows]), np.asarray(white[rows])])
        logits = []
        for booster, voter in zip(boosters, spec['voters']):
            cols = [union.index(name) for name in voter['names'][:-61]] + list(range(439, 500))
            p = np.clip(booster.predict(feats[:, cols].astype(np.float32)).astype(np.float64), 1e-6, 1 - 1e-6)
            logits.append((np.log(p) - np.log1p(-p)).astype(np.float32))
        bag = 1 / (1 + np.exp(-np.mean(np.stack(logits), axis=0, dtype=np.float32).astype(np.float64)))
        bag = np.clip(bag, 1e-6, 1 - 1e-6)
        offline.append(np.log(bag) - np.log1p(-bag))
    golden = np.load(ROOT / 'models/model73/deployment_validation/parity_predictions.npz')['prediction']
    # The golden file was produced on Linux; Windows float32 kernels differ by ~6e-8.
    chain_golden_error = float(np.max(np.abs(np.asarray(chain_predictions) - golden)))
    np.testing.assert_allclose(chain_predictions, golden, rtol=0, atol=1e-6)
    golden = np.asarray(chain_predictions)
    chain = np.log(np.asarray(golden, np.float64)) - np.log1p(-np.asarray(golden, np.float64))
    expected = 1 / (1 + np.exp(-((chain - spec['mu_chain']) / spec['sd_chain']
                                 + spec['weight'] * (np.concatenate(offline) - spec['mu_dist']) / spec['sd_dist'])))
    offline_error = float(np.max(np.abs(np.asarray(predictions) - expected)))
    if offline_error > 2e-5:
        raise AssertionError(f'offline parity error {offline_error}')
    second = []
    for _, _, history, online in golden_sample:
        predictor = SeriesPredictor(bundle, history)
        second.extend(predictor.step(v) for v in online[:min(180, len(online))])
    np.testing.assert_array_equal(predictions, second)
    from main import infer, train
    touched = []

    def stream():
        for i, v in enumerate(golden_sample[0][3][:10]):
            touched.append(i)
            yield v
    output = infer([(golden_sample[0][2], stream())], str(BUNDLE / 'resources'))
    assert next(output) is None and not touched
    for i in range(10):
        assert math.isfinite(next(output)) and len(touched) == i + 1

    class NoData:
        def __iter__(self):
            raise AssertionError('Training data consumed')
    train(NoData(), str(BUNDLE / 'resources'))
    result = {'rows': len(predictions), 'chain_max_error_vs_linux_golden': chain_golden_error,
              'offline_cache_max_error': offline_error, 'deterministic_replay': True,
              'incremental_consumption': True, 'train_does_not_consume_data': True,
              'python': sys.version}
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
    for module in ('solution73', 'solution112'):
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(module,)) as pool:
            results = list(pool.map(timed, sample))
        report[module] = {'workers': workers, 'series': count, 'rows': sum(x['rows'] for x in results),
                          'wall_seconds': time.perf_counter() - started,
                          'worker_seconds': sum(x['seconds'] for x in results)}
    report['ratio_worker_seconds'] = report['solution112']['worker_seconds'] / report['solution73']['worker_seconds']
    report['estimated_cloud_hours'] = 5.62 * report['ratio_worker_seconds']
    write(f'benchmark_w{workers}_n{count}.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('parity', 'benchmark'))
    parser.add_argument('--series', type=int, default=24)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    parity() if args.command == 'parity' else benchmark(args.series, args.workers)
