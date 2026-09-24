"""Model151: privileged teacher features (Model109) for the Model147 re-cut series.

A re-cut is the parent series with the first `shift` online points handed to the history, so the
teacher features can be computed exactly on (history + online[:shift], online[shift:]) and read at the
re-cut's sampled online steps.  Flipped sets (Model150) are not supported (not needed).
Output: models/model147/data/recuts{tag}_teacher.npy  (rows aligned with recuts{tag}.npz)
Usage: python models/model151/teacher_recuts.py [TAG ...]      e.g.  "" _s148 _s149
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from models.model109.teacher_features import NAMES, series_features  # noqa: E402

DATA = ROOT / "models/model147/data"


def build(tag, pool):
    out = DATA / f"recuts{tag}_teacher.npy"
    if out.exists():
        print("exists", out); return
    rec = np.load(DATA / f"recuts{tag}.npz")
    parent, shift, t_on = rec["parent"], rec["shift"], rec["time_online"]
    X = np.empty((len(parent), len(NAMES)), np.float32)
    starts = np.flatnonzero(np.r_[True, (parent[1:] != parent[:-1]) | (shift[1:] != shift[:-1])])
    ends = np.r_[starts[1:], len(parent)]
    t0 = time.time()
    for i, (a, b) in enumerate(zip(starts, ends)):
        h, o = np.asarray(pool[int(parent[a])][0], np.float64), np.asarray(pool[int(parent[a])][1], np.float64)
        s = int(shift[a])
        X[a:b] = series_features(np.r_[h, o[:s]], o[s:])[t_on[a:b]]
        if (i + 1) % 2000 == 0:
            print(f"  {tag or 'base'} {i + 1}/{len(starts)} ({time.time() - t0:.0f}s)", flush=True)
    assert np.isfinite(X).all()
    np.save(out, X)
    print(f"saved {out.name} {X.shape} in {time.time() - t0:.0f}s", flush=True)


def main():
    import models.model111.run_model111 as m111  # installs Linux stubs
    from models.model10.sources import load_development_layout
    from models.model20.features import load_real_pool
    pool = load_real_pool(load_development_layout(ROOT))
    for tag in (sys.argv[1:] or [""]):
        build("" if tag in ("base", '""') else tag, pool)


if __name__ == "__main__":
    main()
