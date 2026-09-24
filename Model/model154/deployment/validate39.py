"""Validate a bundle whose student weight differs from Submission31's 0.5 (Model164: dev-selected 1.5).

Same checks as validate35.py, but the expected values are rebuilt from the saved offline student logits
(models/model152/holdout_student_{TAG}.npz) and Model152's saved chain / GRU components with the bundle's
own weight, so the parity check covers the new weight too.
Usage: python models/model154/deployment/validate39.py [N] [BUNDLE] [TAG]   (defaults 16 submission39 AP1)
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
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else "submission39"
TAG = sys.argv[3] if len(sys.argv) > 3 else "AP1"
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "models/model152"))


def run_bundle(name, datasets):
    bundle = ROOT / "submissions" / name
    for module in [m for m in sys.modules if m.startswith(("solution", "main", "split_tests_numpy", "gru_numpy", "primitive_runtime",
                                                           "feature_pipeline", "compact_", "model73_", "incremental_", "pairing", "detector"))]:
        sys.modules.pop(module)
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
    import offline_holdout as oh
    manifest = json.loads((ROOT / "submissions" / BUNDLE / "resources/manifest.json").read_text())
    ws = float(manifest["model147_student"]["weight"])
    bc = oh.bundle_constants(); const = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    st = np.load(ROOT / f"models/model152/holdout_student_{TAG}.npz")
    chain, gru = np.load(oh.H / "chain.npy"), np.load(oh.H / "gru.npy")
    expected_all = 1 / (1 + np.exp(-((chain - bc["mu_chain"]) / bc["sd_chain"] + ws * (st["dist"] - const["mu_dist"]) / const["sd_dist"]
                                     + bc["weight_gru"] * (gru - bc["mu_gru"]) / bc["sd_gru"])))
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    ids = ref["ids"][:N]
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    datasets, rows = [], 0
    for sid in ids:
        s = frame.loc[int(sid)]
        h = s.loc[s.period == 1, "value"].to_numpy(np.float32); o = s.loc[s.period == 2, "value"].to_numpy(np.float32)
        datasets.append((h, o)); rows += len(o)
    first, seconds_new = run_bundle(BUNDLE, datasets)
    second, _ = run_bundle(BUNDLE, datasets)
    _, seconds31 = run_bundle("submission31", datasets)
    result = {"bundle": BUNDLE, "offline_tag": TAG, "student_weight": ws, "series": int(N), "rows": int(rows),
              "max_abs_diff_vs_offline": float(np.abs(first - expected_all[:rows]).max()),
              "deterministic_replay_max_abs_diff": float(np.abs(first - second).max()),
              "runtime_ratio_vs_submission31": seconds_new / seconds31,
              "estimated_cloud_hours": 4.7 * seconds_new / seconds31,
              "note": "Submission37 (15 voters) was estimated 5.7 h and took 5.6 h in the cloud"}
    (OUT / f"validation_{BUNDLE}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
