"""Validate a built Model154 bundle exactly as the cloud runs it (main.train + main.infer).

On N holdout series (default 16):
  1. main.train() only validates resources;
  2. main.infer() == the offline prediction of the same voters (models/model152/offline_holdout.py
     `score TAG ...` -> holdout_student_{TAG}.npz, built from the streaming-pass inputs) - tests the packaged
     voters, column mapping and blend (the ALL voters are in-sample on the holdout, so this is a CODE check);
  3. a second run is bit-identical (platform: |diff| <= 1e-8);
  4. seconds per row vs Submission31 -> cloud-hours estimate (Submission33 took 6.6 h in the cloud).
Usage: python models/model154/deployment/validate35.py [N] [BUNDLE] [TAG]     (defaults 16 submission35 ALL3)
Writes models/model154/deployment_validation/validation_{BUNDLE}.json.  Never uploads.
"""
import importlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models/model154/deployment_validation"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else "submission35"
TAG = sys.argv[3] if len(sys.argv) > 3 else "ALL3"


def run_bundle(name, datasets):
    bundle = ROOT / "submissions" / name
    for module in [m for m in sys.modules if m.startswith(("solution", "main", "split_tests_numpy", "gru_numpy", "primitive_runtime",
                                                           "feature_pipeline", "compact_", "model73_", "incremental_", "pairing", "detector"))]:
        del sys.modules[module]
    sys.path.insert(0, str(bundle))
    try:
        main = importlib.import_module("main")
        main.train(iter(()), str(bundle / "resources"))
        t0 = time.perf_counter()
        out = [p for p in main.infer(iter(datasets), str(bundle / "resources")) if p is not None]
        return np.asarray(out, np.float64), time.perf_counter() - t0
    finally:
        sys.path.remove(str(bundle))


def main():
    OUT.mkdir(exist_ok=True)
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    ids = ref["ids"][:N]
    offline = np.load(ROOT / f"models/model152/holdout_student_{TAG}.npz")
    assert len(offline["rows"]) == len(ref["step"]), "offline scoring must cover the whole holdout"
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    datasets, rows = [], 0
    for sid in ids:
        s = frame.loc[int(sid)]
        h = s.loc[s.period == 1, "value"].to_numpy(np.float32); o = s.loc[s.period == 2, "value"].to_numpy(np.float32)
        datasets.append((h, o)); rows += len(o)
    expected = offline["final"][:rows].astype(np.float64)
    first, seconds_new = run_bundle(BUNDLE, datasets)
    second, _ = run_bundle(BUNDLE, datasets)
    _, seconds31 = run_bundle("submission31", datasets)
    result = {
        "bundle": BUNDLE, "offline_tag": TAG, "series": int(N), "rows": int(rows),
        "max_abs_diff_vs_offline": float(np.abs(first - expected).max()),
        "deterministic_replay_max_abs_diff": float(np.abs(first - second).max()),
        "runtime_ratio_vs_submission31": seconds_new / seconds31,
        "estimated_cloud_hours": 4.7 * seconds_new / seconds31,
        "note": "Submission33 (same code, 15 voters) was estimated 5.4 h and took 6.6 h in the cloud",
    }
    (OUT / f"validation_{BUNDLE}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
