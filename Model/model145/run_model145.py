"""Model145 runner: does adding the 2025 series (external data) to the training rows help?

Same student, columns (317), soft targets, folds and weights as Model111; only the non-real part of
the training set changes.  External rows carry hard labels (no teacher exists for them), exactly
like the splices.  They are in NO held fold and not in the holdout, so every fold may use all of them.
Arms (share of the training weight given to non-real rows, per (step, label) cell as before):
  X40   external rows replace the splices            (real 60% / external 40%)
  SX40  splices + external pooled                    (real 60% / pooled 40%)
  SX60  splices + external pooled                    (real 40% / pooled 60%)
Usage: python models/model145/run_model145.py ARM [ARM ...] [folds]  |  report
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

import models.model140.run_model140 as m140  # noqa: E402  (report())
from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))
ARMS = {"X40": (False, 0.4), "SX40": (True, 0.4), "SX60": (True, 0.6)}


def run_fold(protocol, layout, real, pool, ext, fold, arm):
    out = RUNS / f"{arm}_fold{fold}.npz"
    if out.exists():
        return
    use_splices, mass = ARMS[arm]
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    ex, ey, et = ext["X_union"][:, columns], ext["target"].astype(np.int8), ext["time_online"].astype(t.dtype)
    if use_splices:
        sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
        ex, ey, et = np.concatenate((sx, ex)), np.r_[sy, ey], np.r_[st, et]
    rw, sw, _ = splice_weights(w, y, t, ey, et, mass)
    saved = np.load(ROOT / f"models/model111/runs/fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * saved["prob"], ey.astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    print(f"{arm} fold {fold}: {len(fit):,} real + {len(ey):,} non-real rows, non-real weight share {sw.sum() / (rw.sum() + sw.sum()):.3f}", flush=True)
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        pred[left:left + len(rows)] = model.predict(real[rows][:, columns])
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held)
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    b = tsauc(layout.target[held], np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs current (Model111) {b:.6f} ({s - b:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    args = sys.argv[1:]
    m140.RUNS = RUNS; m140.HERE = HERE
    if args == ["report"]:
        m140.report(); return
    RUNS.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args if not a.isdigit()] or ["SX40"]
    folds = [int(a) for a in args if a.isdigit()] or [0, 1]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    ext = dict(np.load(HERE / "data/external.npz"))
    for fold in folds:
        for arm in arms:
            run_fold(protocol, layout, real, pool, ext, fold, arm)


if __name__ == "__main__":
    main()
