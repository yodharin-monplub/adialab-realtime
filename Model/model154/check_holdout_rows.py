"""Model154 check: do the holdout-derived training rows (streaming features, holdout teacher targets, holdout
re-cuts) help, out of sample?  For held DEV fold k: train the Model152 FULL recipe on the other 4 dev folds
(arm D4) and the same + all holdout material (arm D4H, exactly the run_all.py code path), score on fold k.
A misalignment bug in the holdout pathway would show up as a loss; the difference also measures the value
of +1,667 series.  (Soft targets of both arms come from the same teachers, so the comparison is fair.)
Usage: python models/model154/check_holdout_rows.py FOLD
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
from models.model152.run_full import all_dev_rows  # noqa: E402
from models.model20.run_model20 import reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, RUNS = HERE / "data", HERE / "runs"
F152 = ROOT / "models/model152/runs"


def fit_eval(protocol, layout, real, pool, rec, hrec, k, with_holdout):
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][k]["seed"]
    names, union = protocol["contract"]["names"][k], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    fit = all_dev_rows(layout, seed); fit = fit[layout.row_fold[fit] != k]
    ft = np.load(F152 / "full_teacher.npz"); soft = ft["prob"][np.searchsorted(ft["rows"], fit)]
    ok = rec["fold"] != k
    dev = np.flatnonzero((layout.row_fold >= 0) & (layout.row_fold != k))
    y, t = layout.target[fit].astype(np.int8), layout.time_online[fit]
    ref_y, ref_t = layout.target[dev].astype(np.int8), layout.time_online[dev]
    xr = real[fit][:, columns]
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    sr = np.asarray(split_real[fit])
    ex, ey, et, es = rec["X_union"][ok][:, columns], rec["target"][ok].astype(np.int8), rec["time_online"][ok], rec["split"][ok]
    if with_holdout:
        h = np.load(DATA / "hold_rows.npz"); hsoft = np.load(RUNS / "hold_teacher.npz")["prob"]
        href = np.load(ROOT / "models/model123/holdout_predictions.npz")
        y, t = np.r_[y, h["target"]].astype(np.int8), np.r_[t, h["time_online"]].astype(t.dtype)
        ref_y, ref_t = np.r_[ref_y, href["target"]].astype(np.int8), np.r_[ref_t, href["step"]].astype(ref_t.dtype)
        soft = np.r_[soft, hsoft]; xr = np.concatenate((xr, h["X_union"][:, columns])); sr = np.concatenate((sr, h["split"]))
        ex = np.concatenate((ex, hrec["X_union"][:, columns])); ey = np.r_[ey, hrec["target"]].astype(np.int8)
        et = np.r_[et, hrec["time_online"]]; es = np.concatenate((es, hrec["split"]))
    w = reference_weights(y, t, ref_y, ref_t)
    sx, sy, st = synthetic_block(protocol, layout, pool, k, names)       # plan k: donors never in fold k
    et = et.astype(t.dtype)
    ex, ey, et = np.concatenate((sx, ex)), np.r_[sy, ey], np.r_[st, et]
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * soft, ey.astype(np.float32)]
    x = np.concatenate((xr, ex))
    extra = np.concatenate([sr, np.load(m147.D144 / f"data_white/syn_fold{k}.npy"), es])
    assert len(extra) == len(x) == len(target) == len(rw) + len(sw)
    keep = np.load(m147.D144 / f"runs_white/E_fold{k}.npz")["columns"]
    x = np.concatenate((x, extra), axis=1)[:, keep]; del extra
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == k)
    pred = np.empty(len(held), np.float32)
    for lo in range(0, len(held), 100_000):
        r = held[lo:lo + 100_000]
        pred[lo:lo + len(r)] = model.predict(np.concatenate((real[r][:, columns], np.asarray(split_real[r])), axis=1)[:, keep])
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    print(f"fold {k} {'D4H (+holdout)' if with_holdout else 'D4'}: {s:.6f} [{time.monotonic() - t0:.0f}s]", flush=True)
    return s


def main():
    k = int(sys.argv[1])
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    rec = dict(np.load(m147.HERE / "data/recuts.npz")); rec["split"] = np.load(m147.HERE / "data/recuts_split.npy")
    hrec = dict(np.load(DATA / "recuts_hold.npz"))
    a = fit_eval(protocol, layout, real, pool, rec, hrec, k, False)
    b = fit_eval(protocol, layout, real, pool, rec, hrec, k, True)
    print(f"fold {k}: holdout material adds {b - a:+.6f}", flush=True)


if __name__ == "__main__":
    main()
