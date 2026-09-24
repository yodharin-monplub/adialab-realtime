"""Model146: train the student as a RANKER grouped by online step (the metric's own structure).

TS-AUC = AUC across series at the same step t, averaged over t with weight n_pos*n_neg.  The current
student is a classifier (cross-entropy on soft targets) whose row weights imitate that; here the same
rows / columns / soft targets are used with a LightGBM ranking objective and query = online step t,
so only within-step order is optimised.
  labels : soft target 0.5*y + 0.5*teacher binned to integers 0..BINS (ranking needs integer gains)
  arms   : lambdarank | rank_xendcg           (columns: the current 317, nothing else changes)
Usage: python models/model146/run_model146.py [lambdarank|rank_xendcg ...] [folds]  |  report
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
BINS = 8
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))
TRUNCATION = int(__import__("os").environ.get("RANK_TRUNCATION", "300"))   # lambdarank pairs only involve the top-k of a query


def run_fold(protocol, layout, real, pool, fold, objective):
    out = RUNS / f"{objective}_fold{fold}.npz"
    if out.exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    rw, sw, _ = splice_weights(w, y, t, sy, st, m111.m84.CONFIG["synthetic_mass"])
    saved = np.load(ROOT / f"models/model111/runs/fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    soft = np.r_[0.5 * y.astype(np.float32) + 0.5 * saved["prob"], sy.astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], sx)); weights = np.r_[rw, sw]; step = np.r_[t, st].astype(np.int64)
    order = np.argsort(step, kind="stable")                      # ranking data must be contiguous per query
    x, soft, weights, step = x[order], soft[order], weights[order], step[order]
    group = np.bincount(step)[np.unique(step)]
    label = np.minimum((soft * (BINS + 1)).astype(np.int32), BINS)
    print(f"{objective} fold {fold}: {len(label):,} rows, {len(group)} queries, largest {group.max()}", flush=True)
    params = dict(m111.m84.CONFIG["arms"]["expressive"])
    model = lgb.LGBMRanker(objective=objective, random_state=seed, n_jobs=THREADS, verbosity=-1, deterministic=True,
                           force_col_wise=True, label_gain=list(range(BINS + 1)), lambdarank_truncation_level=TRUNCATION,
                           **params)   # full-query truncation (1,541) needed > 35 min per fold: stopped 2026-09-20
    model.fit(x, label, group=group, sample_weight=weights)
    del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        pred[left:left + len(rows)] = model.predict(real[rows][:, columns])
    # a ranker's raw score is only ordered within a query; TS-AUC compares within a step, so it is valid as is.
    np.savez(out, prediction=(1 / (1 + np.exp(-pred))).astype(np.float32), raw=pred, held_rows=held)
    model.booster_.save_model(str(RUNS / f"{objective}_fold{fold}.txt"))
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
    b = tsauc(layout.target[held], ref, layout.time_online[held])
    print(f"{objective} fold {fold}: {s:.6f} vs classifier (Model111) {b:.6f} ({s - b:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    arms = [a for a in args if not a.isdigit()] or ["lambdarank"]
    folds = [int(a) for a in args if a.isdigit()] or [0, 1]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for fold in folds:
        for arm in arms:
            run_fold(protocol, layout, real, pool, fold, arm)


if __name__ == "__main__":
    main()
