"""Check real fitted models through Crunch's incremental generator contract."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile
import time

import joblib
import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.model11.features import FEATURE_NAMES, compute_features
from models.model11.serving import infer
from models.model11.pipeline import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=int, choices=(11, 12, 13), default=[11, 12, 13])
    args = parser.parse_args()
    with np.load(ROOT/"models/model5/artifacts/split_assignments.npz") as saved:
        held_series = np.flatnonzero(saved["inner_series_fold"] == 0)
    with np.load(ROOT/"models/artifacts/core_features.metadata.npz") as metadata:
        series_ids = metadata["series_ids"][held_series[[0, len(held_series)//2, -1]]]
    frame = ds.dataset(ROOT/"train/X_train.parquet", format="parquet").to_table(
        filter=ds.field("id").isin(series_ids.tolist())).to_pandas().sort_index()
    records = []
    for series_id in series_ids:
        series = frame.loc[series_id]
        records.append((series.loc[series.period == 1, "value"].to_numpy(), series.loc[series.period == 2, "value"].to_numpy()))
    reports = []
    for number in args.models:
        latest = json.loads((ROOT/f"models/model{number}/runs/latest.json").read_text())
        directory = ROOT/latest["directory"]
        name = "model.joblib" if number == 12 else "model.txt"
        source = directory/("fold0.joblib" if number == 12 else "fold0.txt")
        if number == 12:
            model = joblib.load(source)
            model.n_jobs = 1
        else:
            model = lgb.Booster(model_file=str(source))
        expected = []
        for history, online in records:
            x = compute_features(history, online)
            if number == 12:
                p = model.predict_proba(x)[:, 1]
            else:
                p = model.predict(x, num_threads=1)
                if number == 13:
                    p = expit(p)
            expected.extend(np.clip(p, 1e-7, 1-1e-7))
        with tempfile.TemporaryDirectory(prefix="compact-serving-") as temporary:
            target = Path(temporary)
            shutil.copy2(source, target/name)
            write_json(target/"manifest.json", dict(model=number, feature_names=FEATURE_NAMES))
            consumed = []
            def values(sequence):
                for value in sequence:
                    consumed.append(1)
                    yield value
            started = time.monotonic()
            output = infer([(h, values(o)) for h,o in records], target, number)
            assert next(output) is None and not consumed
            actual = []
            for i in range(len(expected)):
                actual.append(next(output))
                assert len(consumed) == i+1
            try:
                next(output)
            except StopIteration:
                pass
            else:
                raise AssertionError("Serving emitted extra predictions")
            seconds = time.monotonic()-started
        error = float(np.max(np.abs(np.array(expected)-actual)))
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
        report = dict(model=number, held_series=len(records), rows=len(actual), maximum_batch_stream_error=error,
                      streaming_seconds=seconds, milliseconds_per_row=1000*seconds/len(actual),
                      readiness_and_incremental_consumption_passed=True)
        reports.append(report)
        print(json.dumps(report), flush=True)
        write_json(directory/"serving_validation.json", report)
    print(f"PASS: {len(reports)} fitted model serving checks")


if __name__ == "__main__":
    main()
