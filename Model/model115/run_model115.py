"""Model115: Model111 distillation with 3x more real rows (1.2M instead of 400k).

Real fit rows are sampled (seeded) from ALL rows of the three fitting folds
(folds outside {k, k+1}); the legacy 20% hash sampling is dropped.  Soft
targets come from Model111's saved cross-fitted teachers (teacher j, trained
on folds outside {k, k+1, j}, scores the new rows of fold j).  Synthetic rows,
40% synthetic cell mass, weights, 317 fold columns and the expressive
LightGBM recipe are unchanged.

Usage: python models/model115/run_model115.py [fold ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model20.run_model20 import reference_weights, splice_weights, synthetic_tasks  # noqa: E402
from models.model21.run_model21 import read_cache  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
REAL_ROWS = 1_200_000
ALPHA = 0.5
TAG = "a050r12"
THREADS = 8


def teacher_for_rows(layout, real, teacher, columns, fold, rows):
    prob = np.full(len(rows), np.nan, dtype=np.float32)
    for j in range(5):
        if j in (fold, (fold + 1) % 5):
            continue
        booster = lgb.Booster(model_file=str(m111.RUNS / f"fold{fold}_teacher{j}.txt"))
        mask = layout.row_fold[rows] == j
        target_rows = rows[mask]
        out = np.empty(len(target_rows), dtype=np.float32)
        for left in range(0, len(target_rows), 200_000):
            r = target_rows[left:left + 200_000]
            x = np.hstack([real[r][:, columns], np.asarray(teacher[r])])
            out[left:left + len(r)] = booster.predict(x, num_threads=THREADS)
        prob[mask] = out
    if not np.isfinite(prob).all():
        raise ValueError("teacher coverage failed")
    return prob


def run_fold(protocol, layout, real, pool, teacher, fold):
    out = RUNS / f"{TAG}_fold{fold}.npz"
    if out.exists():
        return
    held = np.flatnonzero(layout.row_fold == fold)
    plan = protocol["plans"][fold]
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fitting = (layout.row_fold != fold) & (layout.row_fold != (fold + 1) % 5)
    eligible = np.flatnonzero(fitting)
    fit = np.sort(np.random.default_rng(115_000 + fold).choice(eligible, REAL_ROWS, replace=False))
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    rw, sw, _ = splice_weights(weights, y, t, sy, st, m111.m84.CONFIG["synthetic_mass"])
    parent20 = json.loads((m111.m84.PARENT20 / "protocol.json").read_text())
    sbase = read_cache(m111.m84.PARENT20, f"aligned_25_fold{fold}_features", {
        "protocol_sha256": m111.m84.CONFIG["model20_protocol_sha256"], "rows": len(sy),
        "names": parent20["contract"]["names"][fold], "tasks": len(tasks)})
    swhite = m111.m84.white_source.cached_white(m111.m84.PARENT57 / f"aligned_25_fold{fold}_white.npy",
        m111.m84.CONFIG["model57_protocol_sha256"], len(sy))
    sx = np.concatenate((np.asarray(sbase), np.asarray(swhite)), axis=1)
    del tasks, sbase, swhite
    started = time.monotonic()
    soft = teacher_for_rows(layout, real, teacher, columns, fold, fit)
    print(f"fold {fold}: teacher targets for {len(fit):,} rows ({time.monotonic() - started:.0f}s)", flush=True)
    target = ALPHA * y.astype(np.float32) + (1 - ALPHA) * soft
    x = np.empty((len(fit) + len(sy), len(columns)), dtype=np.float32)
    for left in range(0, len(fit), 200_000):
        r = fit[left:left + 200_000]
        x[left:left + len(r)] = real[r][:, columns]
    x[len(fit):] = sx
    del sx
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS,
                              verbosity=-1, deterministic=True, force_col_wise=True,
                              **m111.m84.CONFIG["arms"]["expressive"])
    started = time.monotonic()
    model.fit(x, np.r_[target, sy.astype(np.float32)], sample_weight=np.r_[rw, sw])
    del x
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        prediction[left:left + len(rows)] = model.predict(real[rows][:, columns])
    model.booster_.save_model(str(RUNS / f"{TAG}_fold{fold}.txt"))
    np.savez(out, prediction=prediction, held_rows=held)
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    print(f"fold {fold}: Model115 raw {score:.6f} ({time.monotonic() - started:.0f}s fit)", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    for fold in folds:
        run_fold(protocol, layout, real, pool, teacher, fold)


if __name__ == "__main__":
    main()
