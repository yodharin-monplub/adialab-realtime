"""Cloud-precision stream parity, shared-work checks, and matched CPU benchmarks."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
import weakref

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pyarrow.dataset as ds
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "submissions/submission17"
BASELINE_RESOURCES = ROOT / "models/model16/submission_validation/baseline_resources"
OUTPUT = ROOT / "models/model17/deployment_validation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUNDLE))


def records(count, full_precision=False):
    with np.load(ROOT / "models/model5/artifacts/split_assignments.npz") as p:
        positions = np.flatnonzero(p["inner_series_fold"] >= 0)
    with np.load(ROOT / "models/artifacts/core_features.metadata.npz") as p:
        ids = p["series_ids"][np.random.default_rng(16312).choice(positions, count, replace=False)]
    frame = ds.dataset(ROOT / "train/X_train.parquet", format="parquet").to_table(
        columns=["id", "time", "value", "period"], filter=ds.field("id").isin(ids.tolist())).to_pandas()
    if frame.index.names != ["id", "time"]:
        frame = frame.set_index(["id", "time"])
    frame = frame.sort_index()
    dtype = np.float64 if full_precision else np.float32
    return [(int(identifier), frame.loc[identifier].query("period == 1").value.to_numpy(dtype=dtype),
             frame.loc[identifier].query("period == 2").value.to_numpy(dtype=dtype)) for identifier in ids]


def parity():
    from solution16 import Bundle as Bundle16, SeriesPredictor as Predictor16, logistic
    from primitive_runtime import _load_bundle, SeriesPredictor as Predictor6
    from solution17 import Bundle, SeriesPredictor
    sample = records(3)
    shared, reference16, reference6 = Bundle(BUNDLE / "resources"), Bundle16(BUNDLE / "resources"), _load_bundle(str(BASELINE_RESOURCES))
    tested = 0
    errors = np.zeros(3)
    for _, history, online in sample:
        online = online[:180]
        first, second = Predictor16(reference16, history), Predictor6(reference6, history)
        expected16 = np.array([first.step(v) for v in online])
        expected6 = np.array([second.step(v) for v in online])
        margin16 = logit(expected16).astype(np.float32)
        margin6 = logit(np.clip(expected6.astype(np.float32), 1e-6, 1-1e-6)).astype(np.float32)
        combined = (.85*margin16.astype(float) + .15*margin6.astype(float)).astype(np.float32)
        expected = logistic(combined.astype(float))
        predictor = SeriesPredictor(shared, history)
        with patch.object(shared.base.detectors, "predict_fold_margins", wraps=shared.base.detectors.predict_fold_margins) as detector_call, \
             patch.object(predictor.frame.detector, "step", wraps=predictor.frame.detector.step) as feature_call:
            actual, branch16, branch6 = [], [], []
            for v in online:
                actual.append(predictor.step(v))
                branch16.append(predictor.last_model16)
                branch6.append(predictor.last_model6)
                assert len(predictor.frame.feature_values) == 3
            assert detector_call.call_count == len(online)
            assert feature_call.call_count == len(online)
        np.testing.assert_array_equal(branch16, expected16)
        np.testing.assert_array_equal(branch6, expected6)
        np.testing.assert_array_equal(actual, expected)
        errors = np.maximum(errors, [np.max(np.abs(np.array(branch16)-expected16)), np.max(np.abs(np.array(branch6)-expected6)), np.max(np.abs(np.array(actual)-expected))])
        # Releasing a series should not leave a self-referential bundle/cache
        # cycle that accumulates historical detector state between GC passes.
        frame_reference = weakref.ref(predictor.frame)
        del predictor, detector_call, feature_call
        assert frame_reference() is None, "Per-series frame retained after release"
        repeated = SeriesPredictor(shared, history)
        np.testing.assert_array_equal(actual, [repeated.step(v) for v in online])
        changed = online.copy(); changed[len(online)//2:] += 100
        future = SeriesPredictor(shared, history)
        np.testing.assert_array_equal(np.array(actual)[:len(online)//2], [future.step(v) for v in changed[:len(online)//2]])
        tested += len(online)
        print(f"Independent branch and shared-stream parity: {tested} observations", flush=True)
    from main import train, infer
    class NoData:
        def __iter__(self): raise AssertionError("Training data were consumed")
    train(NoData(), str(BUNDLE / "resources"))
    consumed = []
    def stream():
        for i, value in enumerate(sample[0][2][:12]):
            consumed.append(i)
            yield value
    output = infer([(sample[0][0], sample[0][1], stream())], str(BUNDLE / "resources"))
    assert next(output) is None and not consumed
    for i in range(12):
        assert np.isfinite(next(output)) and len(consumed) == i + 1
    assert list(output) == []
    result = dict(observations=tested, branch16_max_error=float(errors[0]), branch6_max_error=float(errors[1]),
                  final_max_error=float(errors[2]), deterministic=True, incremental_consumption=True,
                  no_training_data_consumed=True, expensive_detector_calls_per_point=1,
                  history_dtype="float32", online_dtype="float32", python=sys.version)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


_bundle = None
_predictor = None


def initialize(variant):
    global _bundle, _predictor
    if variant == "model17":
        from solution17 import Bundle, SeriesPredictor
    else:
        from solution16 import Bundle, SeriesPredictor
    _bundle, _predictor = Bundle(BUNDLE / "resources"), SeriesPredictor


def timed(record):
    identifier, history, online = record
    start = time.perf_counter()
    predictor = _predictor(_bundle, history)
    values = np.array([predictor.step(v) for v in online], dtype=np.float32)
    return dict(id=identifier, rows=len(online), seconds=time.perf_counter()-start,
                rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), values


def benchmark(variant, count, workers):
    sample = records(count, full_precision=variant == "model16_full_precision")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize, initargs=(variant,)) as pool:
        results = list(pool.map(timed, sample))
    elapsed = time.perf_counter()-start
    metrics = [result[0] for result in results]
    result = dict(variant=variant, workers=workers, series=count, rows=sum(r["rows"] for r in metrics),
                  wall_seconds=elapsed, summed_worker_seconds=sum(r["seconds"] for r in metrics),
                  peak_worker_rss_mib=max(r["rss_mib"] for r in metrics), series_results=metrics,
                  python=sys.version, history_and_online_dtype="float64" if variant == "model16_full_precision" else "float32",
                  note="Includes worker startup; no cloud IPC/scoring overhead, labels, or parent input-data RSS.")
    OUTPUT.mkdir(exist_ok=True)
    name = f"{variant}_w{workers}_n{count}"
    np.savez_compressed(OUTPUT / f"{name}.npz", id=np.array([r["id"] for r in metrics]),
                        lengths=np.array([r["rows"] for r in metrics]), prediction=np.concatenate([r[1] for r in results]))
    (OUTPUT / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "series_results"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("parity", "benchmark"))
    parser.add_argument("--variant", choices=("model16", "model17", "model16_full_precision"), default="model17")
    parser.add_argument("--series", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "parity": parity()
    else: benchmark(args.variant, args.series, args.workers)
