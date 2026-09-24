"""Model155: wider student inputs.  SRE40 recipe (Model147) but every voter sees ALL 500 union columns
(439 detector union of the five fold selections + 61 whitening) + the 27 split tests, instead of its fold's
317 columns + the correlation-pruned split tests.  The serving code maps voters by column NAME
(solution147), so a wider voter needs no serving change.
Arm U40 (re-cut set "").  Compared with SRE40 per dev fold.
Usage: python models/model155/run_model155.py FOLD [FOLD ...]
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model147.run_model147 as m147  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
M111 = ROOT / "models/model111/runs"


def run_fold(protocol, layout, real, pool, rec, fold, arm="U40"):
    out = RUNS / f"{arm}_fold{fold}.npz"
    if out.exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    ok = (rec["fold"] != fold) & (rec["fold"] != (fold + 1) % 5)
    sx317, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    if arm == "U40C":      # clean: all 500 union columns computed for the splices (build_splice_union.py)
        sx = np.load(HERE / f"data/splice_union_fold{fold}.npy"); assert len(sx) == len(sy)
    else:                  # pilot U40: only the fold's 317 columns are cached for splices -> NaN elsewhere
        sx = np.full((len(sy), len(union)), np.nan, np.float32); sx[:, columns] = sx317
    del sx317
    ex, ey, et = np.concatenate((sx, rec["X_union"][ok])), np.r_[sy, rec["target"][ok]].astype(np.int8), \
        np.r_[st, rec["time_online"][ok]].astype(t.dtype)
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    saved = np.load(M111 / f"fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * saved["prob"], ey.astype(np.float32)]
    x = np.concatenate((np.asarray(real[fit]), ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{fold}.npy"), rec["split"][ok]])
    x = np.concatenate((x, extra), axis=1); del extra
    assert x.shape[1] == len(union) + 27 and len(x) == len(target)
    print(f"{arm} fold {fold}: {x.shape[1]} columns, {len(x):,} rows", flush=True)
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=m147.THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), np.float32)
    for lo in range(0, len(held), 100_000):
        r = held[lo:lo + 100_000]
        pred[lo:lo + len(r)] = model.predict(np.concatenate((np.asarray(real[r]), np.asarray(split_real[r])), axis=1))
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held)
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    r = tsauc(layout.target[held], np.load(m147.RUNS / f"SRE40_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs SRE40 {r:.6f} ({s - r:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    arm = next((a for a in sys.argv[1:] if not a.isdigit()), "U40")
    folds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [0, 1]
    RUNS.mkdir(parents=True, exist_ok=True)
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    rec = dict(np.load(m147.HERE / "data/recuts.npz")); rec["split"] = np.load(m147.HERE / "data/recuts_split.npy")
    for f in folds:
        run_fold(protocol, layout, real, pool, rec, f, arm)


if __name__ == "__main__":
    main()
