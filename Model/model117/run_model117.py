"""Model117: Model111 distillation with seed-averaged teachers.

Teacher noise is a plausible limit on distillation gains (Model110 showed
a *stronger* single teacher did not help, but a *less noisy* one is a
different question).  For each held fold k and each fitting fold j, SEEDS
teachers are trained with different LightGBM seeds and column subsamples
and their probabilities averaged before forming the soft target
0.5*y + 0.5*teacher.  Everything else is Model111 (frozen Model84 recipe).

Usage: python models/model117/run_model117.py [fold ...]
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

from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = 3


def seed_averaged_teachers(layout, real, teacher, columns, fold, fit):
    path = m111.RUNS / f"fold{fold}_teacher.npz"
    if path.exists():
        saved = np.load(path)
        assert np.array_equal(saved["rows"], fit)
        return saved["prob"]
    prob = np.full(len(fit), np.nan, dtype=np.float32)
    report = {}
    for j in range(5):
        if j in (fold, (fold + 1) % 5):
            continue
        pool_rows = np.flatnonzero(~np.isin(layout.row_fold, [fold, (fold + 1) % 5, j]))
        mask = layout.row_fold[fit] == j
        target_rows = fit[mask]
        xt = np.hstack([real[target_rows][:, columns], np.asarray(teacher[target_rows])])
        acc = np.zeros(len(target_rows), dtype=np.float64)
        started = time.monotonic()
        for s in range(SEEDS):
            rows = np.sort(np.random.default_rng(117_000 + 100 * fold + 10 * j + s).choice(
                pool_rows, size=min(m111.TEACHER_ROWS, len(pool_rows)), replace=False))
            x = np.hstack([real[rows][:, columns], np.asarray(teacher[rows])])
            y, t = layout.target[rows], layout.time_online[rows]
            params = {**m111.TEACHER_PARAMS, "colsample_bytree": (0.5, 0.4, 0.6)[s]}
            model = lgb.LGBMClassifier(objective="binary", random_state=31 + j + 97 * s,
                                       n_jobs=m111.THREADS, verbosity=-1, force_col_wise=True, **params)
            model.fit(x, y, sample_weight=time_pair_weights(y, t))
            del x
            acc += model.predict_proba(xt)[:, 1]
        prob[mask] = (acc / SEEDS).astype(np.float32)
        report[j] = tsauc(layout.target[target_rows], prob[mask], layout.time_online[target_rows])
        print(f"fold {fold} teacher {j} x{SEEDS}: {report[j]:.5f} on {mask.sum():,} fit rows "
              f"({time.monotonic() - started:.0f}s)", flush=True)
    if not np.isfinite(prob).all():
        raise ValueError("teacher did not cover every fit row")
    np.savez(path, rows=fit, prob=prob)
    (m111.RUNS / f"fold{fold}_teacher.json").write_text(json.dumps(report, indent=2))
    return prob


def main():
    m111.RUNS = HERE / "runs"
    m111.RUNS.mkdir(parents=True, exist_ok=True)
    m111.ALPHA = 0.5
    m111.TAG = "a050s3"
    m111.THREADS = 8
    m111.teacher_probabilities = seed_averaged_teachers
    m111.main()


if __name__ == "__main__":
    main()
