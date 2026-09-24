"""Parity: the 129 GRU inputs built step by step by the Submission34 bundle == the training export rows."""
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "submissions/submission34"
sys.path.insert(0, str(BUNDLE)); sys.path.insert(1, str(ROOT))
import solution148  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
d = np.load(ROOT / "models/model148/data/sequences_split.npz")
X, pos, ids = d["X"], d["series_position"], d["series_ids"]
starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); ends = np.r_[starts[1:], len(pos)]
bundle = solution148.Bundle(BUNDLE / "resources")
frame = pd.read_parquet(ROOT / "train/X_train.parquet")
worst_head = worst_split = 0.0; steps = 0
for a, b in list(zip(starts, ends))[:N]:
    sid = int(ids[pos[a]]); s = frame.loc[sid]
    h = s.loc[s.period == 1, "value"].to_numpy(np.float32); o = s.loc[s.period == 2, "value"].to_numpy(np.float32)
    p = solution148.SeriesPredictor(bundle, h)
    for t, value in enumerate(o):
        value = float(value)
        row = np.asarray(p.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6); np.clip(row, -1e6, 1e6, out=row)
        white = np.asarray(p.white.step(value), dtype=np.float32)
        x = bundle.gru148_input(row, white, p.split.step(value), t)
        ref = X[a + t].astype(np.float32)
        worst_head = max(worst_head, float(np.abs(x[:102] - ref[:102]).max()))
        worst_split = max(worst_split, float(np.abs(x[102:] - ref[102:]).max()))
        steps += 1
print(f"{N} series, {steps} steps: max |diff| first 102 inputs {worst_head:.2e}, 27 split inputs {worst_split:.2e}")
