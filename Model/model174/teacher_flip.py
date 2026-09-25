"""Model174: teacher inputs for a FLIPPED re-cut set (Model150's mirrored sets).

models/model151/teacher_recuts.py says outright that flipped sets are "not supported (not needed)":
it rebuilds each re-cut from the pool WITHOUT mirroring, so its features would describe the unflipped
series while the training rows describe the flipped one. This builds both missing pieces with the mirror
applied exactly as models/model147/build_recuts.py --flip does it - centre = 2*mean(ORIGINAL history),
then history and online become centre - history, centre - online:
  models/model147/data/recuts{tag}_teacher.npy   25 Model109 privileged features
  models/model158/data/recut_oracle{tag}.npy      7 Model157 oracle features
Rows stay aligned with recuts{tag}.npz, so models/model164/run_model164.load_pooled can pool the set.
  python models/model174/teacher_flip.py _f150 [_f151 ...]
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

DATA = ROOT / "models/model147/data"
ORACLE_DIR = ROOT / "models/model158/data"


def rebuild(pool, parent, shift, flip):
    """The re-cut series exactly as build_recuts.py made it (mirrored when flip)."""
    h = np.asarray(pool[int(parent)][0], np.float64)
    o = np.asarray(pool[int(parent)][1], np.float64)
    history, online = np.r_[h, o[:shift]], o[shift:]
    if flip:
        centre = 2.0 * float(h.mean())          # the ORIGINAL history's mean, as in build_recuts.py
        history, online = centre - history, centre - online
    return history, online


def build(tag, pool, flip):
    from models.model109.teacher_features import NAMES, series_features
    from models.model157.oracle_bayes import series_oracle
    rec = np.load(DATA / f"recuts{tag}.npz")
    parent, shift, t_on = rec["parent"], rec["shift"], rec["time_online"]
    starts = np.flatnonzero(np.r_[True, (parent[1:] != parent[:-1]) | (shift[1:] != shift[:-1])])
    ends = np.r_[starts[1:], len(parent)]
    jobs = [(DATA / f"recuts{tag}_teacher.npy", series_features, len(NAMES)),
            (ORACLE_DIR / f"recut_oracle{tag}.npy", series_oracle, 7)]
    for out, fn, width in jobs:
        if out.exists():
            print("exists", out.name, flush=True)
            continue
        X = np.empty((len(parent), width), np.float32)
        t0 = time.time()
        for i, (a, b) in enumerate(zip(starts, ends)):
            history, online = rebuild(pool, parent[a], int(shift[a]), flip)
            X[a:b] = fn(history, online)[t_on[a:b]]
            if (i + 1) % 2000 == 0:
                print(f"  {out.name} {i + 1}/{len(starts)} ({time.time() - t0:.0f}s)", flush=True)
        assert np.isfinite(X).all()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, X)
        print(f"saved {out.name} {X.shape} in {time.time() - t0:.0f}s", flush=True)


def main():
    import models.model111.run_model111  # noqa: F401  (installs the Linux stubs the legacy code wants)
    from models.model10.sources import load_development_layout
    from models.model20.features import load_real_pool
    pool = load_real_pool(load_development_layout(ROOT))
    for tag in (sys.argv[1:] or ["_f150"]):
        flip = tag.startswith("_f")                 # Model150 named every mirrored set _f1xx
        print(f"=== {tag} (flip={flip})", flush=True)
        build(tag, pool, flip)


if __name__ == "__main__":
    main()
