"""Model159 step 1: teacher inputs for every synthetic row of the all-series students, so splices and re-cuts
can get soft targets (Model158) in the final recipe.
  dev re-cuts, sets _s148/_s149 : 7 oracle features -> models/model158/data/recut_oracle{tag}.npy
                                  (set "" already exists as recut_oracle.npy; the 25 Model109 features exist as
                                  models/model147/data/recuts{tag}_teacher.npy from Model151)
  holdout re-cuts, all 3 sets   : 25 Model109 + 7 oracle features -> models/model154/data/recuts_hold{tag}_teacher.npy
Usage: python models/model159/prep_soft.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from models.model109.teacher_features import series_features as tf109  # noqa: E402
from models.model157.oracle_bayes import series_oracle  # noqa: E402


def dev_recut_oracle(pool, tag):
    out = ROOT / f"models/model158/data/recut_oracle{tag}.npy"
    if out.exists():
        return
    rec = np.load(ROOT / f"models/model147/data/recuts{tag}.npz")
    parent, shift, t_on = rec["parent"], rec["shift"], rec["time_online"]
    X = np.empty((len(parent), 7), np.float32)
    st = np.flatnonzero(np.r_[True, (parent[1:] != parent[:-1]) | (shift[1:] != shift[:-1])]); en = np.r_[st[1:], len(parent)]
    for a, b in zip(st, en):
        h, o = np.asarray(pool[int(parent[a])][0], np.float64), np.asarray(pool[int(parent[a])][1], np.float64)
        s = int(shift[a]); X[a:b] = series_oracle(np.r_[h, o[:s]], o[s:])[t_on[a:b]]
    np.save(out, X); print("saved", out.name, X.shape, flush=True)


def holdout_recut_teacher(frame, tag):
    out = ROOT / f"models/model154/data/recuts_hold{tag}_teacher.npy"
    if out.exists():
        return
    rec = np.load(ROOT / f"models/model154/data/recuts_hold{tag}.npz")
    sid, shift, t_on = rec["sid"], rec["shift"], rec["time_online"]
    X = np.empty((len(sid), 32), np.float32)
    st = np.flatnonzero(np.r_[True, (sid[1:] != sid[:-1]) | (shift[1:] != shift[:-1])]); en = np.r_[st[1:], len(sid)]
    for a, b in zip(st, en):
        s = frame.loc[int(sid[a])]
        h = s.loc[s.period == 1, "value"].to_numpy(np.float64); o = s.loc[s.period == 2, "value"].to_numpy(np.float64)
        k = int(shift[a]); hh, oo = np.r_[h, o[:k]], o[k:]
        X[a:b] = np.hstack([tf109(hh, oo), series_oracle(hh, oo)])[t_on[a:b]]
    np.save(out, X); print("saved", out.name, X.shape, flush=True)


def main():
    import pandas as pd
    import models.model111.run_model111  # noqa: F401  (Linux stubs)
    from models.model10.sources import load_development_layout
    from models.model20.features import load_real_pool
    t0 = time.time()
    pool = load_real_pool(load_development_layout(ROOT))
    for tag in ("_s148", "_s149"):
        dev_recut_oracle(pool, tag)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    for tag in ("", "_s148", "_s149"):
        holdout_recut_teacher(frame, tag)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
