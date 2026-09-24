"""Model156 runner: SRE40 (Model147) + the 3 calibrated Bayesian evidence columns (bayes_evidence.py),
appended after the usual column selection (never pruned).  One change vs SRE40 -> compared per dev fold.
Arm BE40 (re-cut set "").  Usage: python models/model156/run_model156.py FOLD [FOLD ...]
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
DATA, RUNS = HERE / "data", HERE / "runs"
M111 = ROOT / "models/model111/runs"


def run_fold(protocol, layout, real, pool, rec, fold, arm="BE40"):
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
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    ex, ey, et = np.concatenate((sx, rec["X_union"][ok][:, columns])), np.r_[sy, rec["target"][ok]].astype(np.int8), \
        np.r_[st, rec["time_online"][ok]].astype(t.dtype)
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    saved = np.load(M111 / f"fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * saved["prob"], ey.astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{fold}.npy"), rec["split"][ok]])
    keep = np.load(m147.D144 / f"runs_white/E_fold{fold}.npz")["columns"]
    ev_real = np.load(DATA / "real.npy", mmap_mode="r")
    ev = np.concatenate([np.asarray(ev_real[fit]), np.load(DATA / f"syn_fold{fold}.npy"), np.load(DATA / "recuts.npy")[ok]])
    x = np.concatenate((np.concatenate((x, extra), axis=1)[:, keep], ev), axis=1); del extra, ev
    assert len(x) == len(target)
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=m147.THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    gain = model.booster_.feature_importance("gain"); share = gain[-3:].sum() / gain.sum()
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), np.float32)
    for lo in range(0, len(held), 100_000):
        r = held[lo:lo + 100_000]
        xh = np.concatenate((real[r][:, columns], np.asarray(split_real[r])), axis=1)[:, keep]
        pred[lo:lo + len(r)] = model.predict(np.concatenate((xh, np.asarray(ev_real[r])), axis=1))
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held, columns=keep)
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    ref = tsauc(layout.target[held], np.load(m147.RUNS / f"SRE40_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs SRE40 {ref:.6f} ({s - ref:+.6f}); evidence gain share {share:.3f} "
          f"[{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    folds = [int(a) for a in sys.argv[1:]] or [0, 1]
    RUNS.mkdir(parents=True, exist_ok=True)
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    rec = dict(np.load(m147.HERE / "data/recuts.npz")); rec["split"] = np.load(m147.HERE / "data/recuts_split.npy")
    for f in folds:
        run_fold(protocol, layout, real, pool, rec, f)


if __name__ == "__main__":
    main()
