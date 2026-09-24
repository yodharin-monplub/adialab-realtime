"""Model149 runner: the Submission33 student recipe (splices + one re-cut set, 40%) with the 30 evidence
columns added to the 317 + 27 split-test columns; near-duplicate columns (|rho| > .95) pruned per fold.

Arms: EV40 (re-cut set ""), EV40b ("_s148"), EV40c ("_s149") - three students for a bag, exactly like
SRE40 / SRE40b / SRE40c of Model147, so BAG(EV40*) vs BAG3 changes one thing: the evidence columns.
Usage: python models/model149/run_model149.py ARM [ARM ...] [folds]
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
D144, D147, D149 = ROOT / "models/model144/data_white", ROOT / "models/model147/data", HERE / "data"
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))
SETS = {"EV40": "", "EV40b": "_s148", "EV40c": "_s149"}


def prune(x, fold, thr=0.95):
    rng = np.random.default_rng(fold)
    rows = rng.choice(len(x), 200_000, replace=False)
    c = np.nan_to_num(np.abs(np.corrcoef(x[rows].astype(np.float64), rowvar=False)))
    keep = []
    for j in range(c.shape[0]):
        if all(c[j, i] <= thr for i in keep):
            keep.append(j)
    return np.asarray(keep)


def run_fold(protocol, layout, real, pool, fold, arm):
    out = RUNS / f"{arm}_fold{fold}.npz"
    if out.exists():
        return
    t0 = time.monotonic(); tag = SETS[arm]
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    rec = np.load(D147 / f"recuts{tag}.npz")
    ok = (rec["fold"] != fold) & (rec["fold"] != (fold + 1) % 5)
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    ey, et = np.r_[sy, rec["target"][ok].astype(np.int8)], np.r_[st, rec["time_online"][ok].astype(t.dtype)]
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    saved = np.load(ROOT / f"models/model111/runs/fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * saved["prob"], ey.astype(np.float32)]
    split_real = np.load(D144 / "real.npy", mmap_mode="r"); ev_real = np.load(D149 / "real.npy", mmap_mode="r")
    x = np.concatenate((
        np.concatenate((real[fit][:, columns], np.asarray(split_real[fit]), np.asarray(ev_real[fit])), axis=1),
        np.concatenate((sx, np.load(D144 / f"syn_fold{fold}.npy"), np.load(D149 / f"syn_fold{fold}.npy")), axis=1),
        np.concatenate((rec["X_union"][ok][:, columns], np.load(D147 / f"recuts{tag}_split.npy")[ok], np.load(D149 / f"recuts{tag}.npy")[ok]), axis=1)))
    assert len(x) == len(target)
    keep = prune(x, fold)
    print(f"{arm} fold {fold}: {x.shape[1]} columns -> {len(keep)} kept ({int((keep >= 344).sum())} of the 30 evidence, "
          f"{int(((keep >= 317) & (keep < 344)).sum())} of the 27 split)", flush=True)
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x[:, keep], target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        xh = np.concatenate((real[rows][:, columns], np.asarray(split_real[rows]), np.asarray(ev_real[rows])), axis=1)[:, keep]
        pred[left:left + len(rows)] = model.predict(xh)
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held, columns=keep)
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    b = tsauc(layout.target[held], np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"], layout.time_online[held])
    same = "SRE40" + arm[4:]
    r = tsauc(layout.target[held], np.load(ROOT / f"models/model147/runs/{same}_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} | vs current student {s - b:+.6f} | vs {same} (same data, no evidence columns) {s - r:+.6f} "
          f"[{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    arms = [a for a in args if not a.isdigit()] or ["EV40"]
    folds = [int(a) for a in args if a.isdigit()] or [0, 1]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for arm in arms:
        for fold in folds:
            run_fold(protocol, layout, real, pool, fold, arm)


if __name__ == "__main__":
    main()
