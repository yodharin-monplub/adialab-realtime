"""Model157 probe: does the oracle teacher improve with more capacity / rows?  One teacher slot only
(held fold 0, teacher j=2: trained on folds 3,4 [outside {0,1,2}], scored on the fit rows of fold 2), same
inputs as teacherB (317 student columns + 25 old + 7 oracle features).  Prints TS-AUC per variant.
Usage: python models/model157/teacher_probe.py
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

from models.model20.run_model20 import fitting_rows  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

THREADS = int(__import__("os").environ.get("LGBM_THREADS", "2"))
BASE = dict(m147.m111.TEACHER_PARAMS)
VARIANTS = {
    "base 450k": (450_000, BASE),
    "900k rows": (900_000, BASE),
    "1500 trees lr.03": (450_000, dict(BASE, n_estimators=1500, learning_rate=0.03)),
    "127 leaves": (450_000, dict(BASE, num_leaves=127, min_child_samples=200)),
}


def main():
    _, protocol, layout, real, pool, old = m147.m111.load_everything()
    new = np.load(ROOT / "models/model157/data/real.npy", mmap_mode="r")
    fold, j = 0, 2
    fit, _ = fitting_rows(layout, fold, protocol["contract"]["recipes"][fold]["seed"])
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in protocol["contract"]["names"][fold]])
    tr = fit[layout.row_fold[fit] == j]
    xt = np.hstack([real[tr][:, columns], np.asarray(old[tr]), np.asarray(new[tr])])
    pool_rows = np.flatnonzero(~np.isin(layout.row_fold, [fold, (fold + 1) % 5, j]))
    for name, (n, params) in VARIANTS.items():
        t0 = time.monotonic()
        rows = np.sort(np.random.default_rng(7000 + 10 * fold + j).choice(pool_rows, min(n, len(pool_rows)), replace=False))
        x = np.hstack([real[rows][:, columns], np.asarray(old[rows]), np.asarray(new[rows])])
        y, t = layout.target[rows], layout.time_online[rows]
        model = lgb.LGBMClassifier(objective="binary", random_state=31 + j, n_jobs=THREADS, verbosity=-1, force_col_wise=True, **params)
        model.fit(x, y, sample_weight=time_pair_weights(y, t)); del x
        p = model.predict_proba(xt)[:, 1]
        print(f"{name:18s} ({len(rows):,} rows): TS-AUC {tsauc(layout.target[tr], p, layout.time_online[tr]):.5f} "
              f"[{time.monotonic() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
