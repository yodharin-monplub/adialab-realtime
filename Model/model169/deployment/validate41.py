"""Validate Submission41 (new GRU part) exactly as the cloud runs it.

Expected values are rebuilt offline from Model152's saved chain, the bundle's student logits
(models/model152/holdout_student_AP3.npz) and the NEW GRU logits (models/model169/data/h256_logits.npy,
first five = one seed x five folds), with the bundle's own constants - so the parity check covers the new
GRU weights, their layer count and their z-score constants.
Usage: python models/model169/deployment/validate41.py [N] [BUNDLE]     (defaults 16 submission41)
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
OUT = ROOT / "models/model169/deployment_validation"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else "submission41"
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
    OUT.mkdir(parents=True, exist_ok=True)
    import offline_holdout as oh
    manifest = json.loads((ROOT / "submissions" / BUNDLE / "resources/manifest.json").read_text())
    ws = float(manifest["model147_student"]["weight"])
    gspec = manifest["model124_gru"]
    bc = oh.bundle_constants(); const = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    st = np.load(ROOT / "models/model152/holdout_student_AP3.npz")
    chain = np.load(oh.H / "chain.npy")
    gru = np.load(ROOT / "models/model169/data/h256_logits.npy")[:5].mean(0)
    expected_all = 1 / (1 + np.exp(-((chain - bc["mu_chain"]) / bc["sd_chain"]
                                     + ws * (st["dist"] - const["mu_dist"]) / const["sd_dist"]
                                     + float(gspec["weight_gru"]) * (gru - float(gspec["mu_gru"])) / float(gspec["sd_gru"]))))
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    datasets, rows = [], 0
    for sid in ref["ids"][:N]:
        s = frame.loc[int(sid)]
        h = s.loc[s.period == 1, "value"].to_numpy(np.float32); o = s.loc[s.period == 2, "value"].to_numpy(np.float32)
        datasets.append((h, o)); rows += len(o)
    first, seconds_new = run_bundle(BUNDLE, datasets)
    second, _ = run_bundle(BUNDLE, datasets)
    _, seconds40 = run_bundle("submission40", datasets)
    result = {"bundle": BUNDLE, "student_weight": ws, "gru_nets": len(gspec["folds"]), "series": int(N), "rows": int(rows),
              "max_abs_diff_vs_offline": float(np.abs(first - expected_all[:rows]).max()),
              "deterministic_replay_max_abs_diff": float(np.abs(first - second).max()),
              "runtime_ratio_vs_submission40": seconds_new / seconds40,
              "estimated_cloud_hours": 4.7 * seconds_new / seconds40,
              "note": "Submission40 took 4.7 h in the cloud (run 119771)"}
    (OUT / f"validation_{BUNDLE}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
