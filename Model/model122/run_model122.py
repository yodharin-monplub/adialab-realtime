"""Model122: seed-bag of the distilled student (Model111 recipe, extra seeds).

Trains the Model111 student (same targets, rows, columns, weights, recipe)
with two additional LightGBM seeds and reports the raw component of the
three-seed logit average (Model111 + 2 new) against Model111 alone.
Student-side variance reduction, complementary to Model117 (teacher side).

The recipe seed must stay fixed because it also selects the frozen fit rows;
only LightGBM's random_state is offset, through a shim on Model111's `lgb`.

Usage: python models/model122/run_model122.py [fold ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
EXTRA_SEEDS = (1001, 2002)


class SeedShim:
    """Forward to lightgbm but offset random_state for the student regressor."""
    offset = 0

    @staticmethod
    def LGBMRegressor(**kw):
        return lgb.LGBMRegressor(**{**kw, "random_state": int(kw["random_state"]) + SeedShim.offset})

    LGBMClassifier = lgb.LGBMClassifier
    Booster = lgb.Booster


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    runs = HERE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        source = m111.RUNS / f"fold{fold}_teacher.npz"
        if source.exists() and not (runs / source.name).exists():
            shutil.copy2(source, runs / source.name)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    m111.RUNS = runs
    m111.ALPHA = 0.5
    m111.THREADS = 8
    m111.lgb = SeedShim
    report = {}
    for fold in folds:
        preds = [np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]]
        for s in EXTRA_SEEDS:
            m111.TAG = f"a050seed{s}"
            SeedShim.offset = s
            preds.append(m111.run_fold(protocol, layout, real, pool, teacher, fold))
        SeedShim.offset = 0
        held = layout.row_fold == fold
        y, t = layout.target[held], layout.time_online[held]
        bag = 1 / (1 + np.exp(-np.mean([lg(p) for p in preds], axis=0)))
        np.savez(runs / f"a050bag3_fold{fold}.npz", prediction=bag.astype(np.float32),
                 held_rows=np.flatnonzero(held))
        report[fold] = {"model111": tsauc(y, preds[0], t), "bag3": tsauc(y, bag, t),
                        "seeds": [tsauc(y, p, t) for p in preds[1:]]}
        print(f"fold {fold}: Model111 {report[fold]['model111']:.6f}  seeds {np.round(report[fold]['seeds'], 6).tolist()}"
              f"  bag3 {report[fold]['bag3']:.6f}", flush=True)
    (runs / "pilot_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
