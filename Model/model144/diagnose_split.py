"""Diagnostic only (uses tau, never feeds a model): how close is the estimated split to the true break?"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from models.model144.split_tests import NAMES  # noqa: E402

L = load_development_layout(ROOT)
from models.model144.split_tests import OUT  # noqa: E402  (data or data_white, by SP_WHITEN)
X = np.load(OUT / "real.npy", mmap_mode="r")
pos = L.series_position
starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); ends = np.r_[starts[1:], L.row_count]
col_level, col_diff = NAMES.index("sp_khat_rel"), NAMES.index("spd_khat_rel")
delays = (8, 16, 32, 64, 128, 256)
err_level = {d: [] for d in delays}; err_diff = {d: [] for d in delays}
for a, b in zip(starts, ends):
    y = L.target[a:b]; n = b - a
    if y.max() == 0:
        continue
    tau = int(np.argmax(y))
    for d in delays:
        t = tau + d
        if t < n:
            err_level[d].append(abs(float(X[a + t, col_level]) * (t + 1) - tau))
            err_diff[d].append(abs(float(X[a + t, col_diff]) * (t + 1) - tau))
print("points after break | median |k_hat - tau| level / diff scan | share within 10 steps level / diff | series")
for d in delays:
    e1, e2 = np.array(err_level[d]), np.array(err_diff[d])
    print(f"{d:5d}   {np.median(e1):6.0f} / {np.median(e2):6.0f}     {np.mean(e1 <= 10):.2f} / {np.mean(e2 <= 10):.2f}    {len(e1)}")
