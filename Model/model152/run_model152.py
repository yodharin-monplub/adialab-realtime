"""Model152: learning curve of the SRE40 student in the number of SERIES.

Question: how much would a final refit on all 10,000 series (instead of fold voters on 3/5 of the
8,333 development series) gain?  Measured by the opposite move: for held fold k, also drop the fitting
fold j = (k+2) % 5 (its real rows, its teacher targets, its re-cuts), i.e. 3/5 -> 2/5 of the series,
and compare with SRE40 on the same held fold.  Splices are kept (their donors are pairs, a few use fold j),
so the measured loss slightly UNDER-states the series effect.
Arm LC2 (SRE40 recipe, two fitting folds).  Usage: python models/model152/run_model152.py [folds]
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


def run_fold(protocol, layout, real, pool, rec, fold, arm="LC2"):
    out = RUNS / f"{arm}_fold{fold}.npz"
    if out.exists():
        return
    t0 = time.monotonic()
    drop = (fold + 2) % 5
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit_all, fitting = fitting_rows(layout, fold, seed)
    saved = np.load(M111 / f"fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit_all)
    kept = layout.row_fold[fit_all] != drop
    fit, soft = fit_all[kept], saved["prob"][kept]
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    ok = ~np.isin(rec["fold"], [fold, (fold + 1) % 5, drop])
    ex, ey, et = rec["X_union"][ok][:, columns], rec["target"][ok].astype(np.int8), rec["time_online"][ok].astype(t.dtype)
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    ex, ey, et = np.concatenate((sx, ex)), np.r_[sy, ey], np.r_[st, et]
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * soft, ey.astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{fold}.npy"), rec["split"][ok]])
    assert len(extra) == len(x) == len(target)
    keep_cols = np.load(m147.D144 / f"runs_white/E_fold{fold}.npz")["columns"]
    x = np.concatenate((x, extra), axis=1)[:, keep_cols]; del extra
    print(f"{arm} fold {fold} (drop {drop}): {len(fit):,} real rows of {len(fit_all):,}, {int(ok.sum()):,} re-cut rows", flush=True)
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=m147.THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        xh = np.concatenate((real[rows][:, columns], np.asarray(split_real[rows])), axis=1)[:, keep_cols]
        pred[left:left + len(rows)] = model.predict(xh)
    np.savez(out, prediction=pred, held_rows=held)
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    r = tsauc(layout.target[held], np.load(m147.RUNS / f"SRE40_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs SRE40 (3 fitting folds) {r:.6f} ({s - r:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    folds = [int(a) for a in sys.argv[1:]] or [0]
    RUNS.mkdir(parents=True, exist_ok=True)
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    rec = dict(np.load(m147.HERE / "data/recuts.npz")); rec["split"] = np.load(m147.HERE / "data/recuts_split.npy")
    for fold in folds:
        run_fold(protocol, layout, real, pool, rec, fold)


if __name__ == "__main__":
    main()
