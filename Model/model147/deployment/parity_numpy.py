"""Parity: numpy serving features (split_tests_numpy.SplitTests) vs the numba batch code that built the
training rows (models/model144/split_tests.py, SP_WHITEN=1).  Usage: python parity_numpy.py [n_series]"""
import os
from pathlib import Path
import sys
import time

os.environ["SP_WHITEN"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.model144.split_tests import series_features  # noqa: E402
from split_tests_numpy import NAMES, SplitTests  # noqa: E402

n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 12
frame = pd.read_parquet(ROOT / "train/X_train.parquet")
ids = frame.index.get_level_values(0).unique()[:n_series]
worst = np.zeros(len(NAMES)); steps = 0; t0 = time.time(); flips = 0
for sid in ids:
    s = frame.loc[sid]; v = s["value"].to_numpy(np.float64); p = s["period"].to_numpy()
    h, o = v[p == 1], v[p == 2]
    ref = series_features(h, o)
    st = SplitTests(h)
    got = np.stack([st.step(x) for x in o])
    scale = np.abs(ref).max(axis=0) + 1e-6
    worst = np.maximum(worst, (np.abs(got - ref) / scale).max(axis=0)); steps += len(o)
    flips += int((got[:, NAMES.index("sp_khat_rel")] != ref[:, NAMES.index("sp_khat_rel")]).sum())
print(f"{len(ids)} series, {steps} steps, {1000 * (time.time() - t0) / steps:.2f} ms/step incl. numba reference")
for n, w in zip(NAMES, worst):
    print(f"  {n:16s} worst relative difference {w:.1e}")
print("steps where the estimated split differs:", flips)
