"""Check Model73 resource parity, causal streaming, and CPU/RAM timing."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / 'submissions/submission27'
OUTPUT = ROOT / 'models/model73/deployment_validation'
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    OUTPUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def records(count):
    with np.load(ROOT / 'models/model5/artifacts/split_assignments.npz') as partition:
        positions = np.flatnonzero(partition['inner_series_fold'] >= 0)
    with np.load(ROOT / 'models/artifacts/core_features.metadata.npz') as meta:
        ids = meta['series_ids'][np.random.default_rng(16312).choice(positions, count, replace=False)]
        total = int(meta['online_lengths'].sum())
    frame = ds.dataset(ROOT / 'train/X_train.parquet', format='parquet').to_table(
        filter=ds.field('id').isin(ids.tolist())).to_pandas().sort_index()
    output = []
    for identifier in ids:
        series = frame.loc[identifier]
        output.append((int(identifier),
                       series.loc[series.period == 1, 'value'].to_numpy(dtype=np.float32),
                       series.loc[series.period == 2, 'value'].to_numpy(dtype=np.float32)))
    return output, total


def parity():
    from solution26 import Bundle as Model26Bundle, logistic, logit
    from solution73 import Bundle, SeriesPredictor
    from model73_whitening import whitening_features
    from feature_pipeline import compute_features_offline
    from compact_features import compute_features
    from primitive_runtime import _bag_probability_rows
    candidate = Bundle(BUNDLE / 'resources')
    base = Model26Bundle(BUNDLE / 'resources')
    sample, _ = records(3)
    golden = []
    max_feature = max_output = max_white = 0.0
    consumed = 0
    for _, history, full_online in sample:
        online = full_online[:180]
        detector_rows = compute_features_offline(history, online, candidate.banks).astype(np.float32)
        np.nan_to_num(detector_rows, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(detector_rows, -1e6, 1e6, out=detector_rows)
        white_rows = whitening_features(history, online)
        for index in (0, 15, 63, 127, 179):
            if index < len(online):
                prefix = whitening_features(history, online[:index + 1])[-1]
                max_white = max(max_white, float(np.max(np.abs(prefix - white_rows[index]))))
                np.testing.assert_array_equal(prefix, white_rows[index])
        base_primitive = np.stack([base.primitive_probabilities(row) for row in detector_rows])
        white_predictions = []
        for model, columns in candidate.white_voters:
            features = np.concatenate((detector_rows[:, columns], white_rows), axis=1)
            white_predictions.append(model.predict(features, num_threads=1).astype(np.float32))
        base_primitive[:, candidate.detectors.bank_count + candidate.core_index] = (
            _bag_probability_rows(np.column_stack(white_predictions)))
        exported = np.stack([candidate.primitive_probabilities(row, white)
                             for row, white in zip(detector_rows, white_rows)])
        max_feature = max(max_feature, float(np.max(np.abs(base_primitive - exported))))
        np.testing.assert_allclose(exported, base_primitive, atol=1e-7, rtol=0)
        primitive_margin = logit(base_primitive).astype(np.float32)
        margin9 = (primitive_margin.astype(np.float64) @ candidate.coefficient).astype(np.float32)
        p9 = logistic(margin9.astype(float)).astype(np.float32)
        m9 = logit(p9.astype(float), epsilon=1e-7)
        compact_rows = compute_features(history, online)
        compact = _bag_probability_rows(np.column_stack([
            model.predict(compact_rows, num_threads=1) for model in candidate.compact_models]))
        m11 = logit(compact).astype(np.float32)
        expected = []
        for index in range(len(online)):
            corrected = np.float32(m9[index] + .1 * (
                np.max(m9[max(0, index - 127):index + 1]) - m9[index]))
            expected.append(float(logistic(float(np.float32(.9 * corrected + .1 * m11[index])))))
        predictor = SeriesPredictor(candidate, history)
        actual = np.asarray([predictor.step(value) for value in online])
        max_output = max(max_output, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
        second = SeriesPredictor(candidate, history)
        np.testing.assert_array_equal(actual, [second.step(value) for value in online])
        golden.append(actual)
        consumed += len(online)
        print(f'Model73 parity: {consumed} observations', flush=True)
    from main import infer, train
    touched = []
    def online_points():
        for index, value in enumerate(sample[0][2][:10]):
            touched.append(index)
            yield value
    outputs = infer([(sample[0][1], online_points())], str(BUNDLE / 'resources'))
    assert next(outputs) is None and not touched
    for index in range(10):
        assert np.isfinite(next(outputs)) and len(touched) == index + 1
    try:
        next(outputs)
    except StopIteration:
        pass
    else:
        raise AssertionError('Extra inference output')
    class NoTrainingData:
        def __iter__(self):
            raise AssertionError('Cloud training data were consumed')
    train(NoTrainingData(), str(BUNDLE / 'resources'))
    result = dict(observations=consumed, whitening_max_error=max_white,
                  primitive_max_error=max_feature, final_probability_max_error=max_output,
                  repeated_inference_bitwise_identical=True, incremental_consumption_passed=True,
                  train_does_not_consume_data=True, python=sys.version)
    np.savez_compressed(OUTPUT / 'parity_predictions.npz', prediction=np.concatenate(golden))
    write_json(OUTPUT / 'parity.json', result)
    print(json.dumps(result, indent=2), flush=True)


def checkpoint_parity():
    """Confirm the exported LightGBM files reproduce frozen held-fold scores."""
    from models.model10.sources import load_development_layout
    from solution73 import Bundle
    protocol = json.loads((ROOT / 'models/model62/runs/6a463687884d27c4/protocol.json').read_text())
    layout = load_development_layout(ROOT)
    base = np.load(ROOT / 'models/model20/runs/82abe027caea808d/real_features.npy', mmap_mode='r')
    white = np.load(ROOT / 'models/model57/runs/dfab0ede1c3206a4/real_white.npy', mmap_mode='r')
    union = protocol['contract']['union_names']
    bundle = Bundle(BUNDLE / 'resources')
    errors = []
    for fold, (model, _) in enumerate(bundle.white_voters):
        held = np.flatnonzero(layout.row_fold == fold)
        positions = np.linspace(0, len(held) - 1, 100, dtype=np.int64)
        rows = held[positions]
        combined = np.concatenate((np.asarray(base[rows]), np.asarray(white[rows])), axis=1)
        columns = [union.index(name) for name in protocol['contract']['names'][fold]]
        prediction = model.predict(combined[:, columns], num_threads=1)
        frozen = np.load(ROOT / f'models/model62/runs/6a463687884d27c4/'
                         f'lightgbm_aligned_40_fold{fold}.npz')['prediction'][positions]
        error = float(np.max(np.abs(prediction - frozen)))
        if error > 1e-7:
            raise AssertionError(f'Model73 fold {fold} checkpoint mismatch: {error}')
        errors.append(error)
    result = dict(rows_per_fold=100, maximum_error=max(errors), fold_max_errors=errors)
    write_json(OUTPUT / 'checkpoint_parity.json', result)
    print(json.dumps(result, indent=2), flush=True)


_worker_bundle = None
_worker_baseline = False


def initialize_worker(baseline=False):
    global _worker_bundle, _worker_baseline
    _worker_baseline = baseline
    if baseline:
        from solution26 import Bundle
        _worker_bundle = Bundle(ROOT / 'submissions/submission26/resources')
    else:
        from solution73 import Bundle
        _worker_bundle = Bundle(BUNDLE / 'resources')


def timed_series(record):
    if _worker_baseline:
        from solution26 import SeriesPredictor
    else:
        from solution73 import SeriesPredictor
    identifier, history, online = record
    started = time.perf_counter()
    predictor = SeriesPredictor(_worker_bundle, history)
    values = np.asarray([predictor.step(value) for value in online], dtype=np.float32)
    return dict(id=identifier, rows=len(online), seconds=time.perf_counter() - started,
                max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                checksum=float(values.astype(float).sum()))


def benchmark(count, workers, baseline=False):
    sample, total = records(count)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(baseline,)) as pool:
        results = list(pool.map(timed_series, sample))
    report = dict(workers=workers, series=count, rows=sum(item['rows'] for item in results),
                  wall_seconds=time.perf_counter() - started,
                  summed_worker_seconds=sum(item['seconds'] for item in results),
                  peak_worker_rss_mib=max(item['max_rss_mib'] for item in results),
                  training_online_rows_reference=total, series_results=results, python=sys.version)
    prefix = 'baseline_' if baseline else ''
    write_json(OUTPUT / f'{prefix}benchmark_w{workers}_n{count}.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'series_results'},
                     indent=2), flush=True)


def cloud_smoke():
    from solution73 import Bundle, SeriesPredictor
    sample, _ = records(3)
    bundle = Bundle(BUNDLE / 'resources')
    prediction = []
    for _, history, online in sample:
        predictor = SeriesPredictor(bundle, history)
        prediction.extend(predictor.step(value) for value in online[:180])
    expected = np.load(OUTPUT / 'parity_predictions.npz')['prediction']
    error = float(np.max(np.abs(np.asarray(prediction) - expected)))
    np.testing.assert_allclose(prediction, expected, atol=1e-7, rtol=0)
    result = dict(observations=len(prediction), maximum_error=error,
                  python=sys.version, cloud_python_compatible=True)
    write_json(OUTPUT / 'cloud_smoke.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('parity', 'checkpoint', 'benchmark', 'cloud-smoke'))
    parser.add_argument('--series', type=int, default=24)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    if args.command == 'parity':
        parity()
    elif args.command == 'checkpoint':
        checkpoint_parity()
    elif args.command == 'cloud-smoke':
        cloud_smoke()
    else:
        benchmark(args.series, args.workers, args.baseline)
