"""Model161 step 1: GRU soft targets from the oracle teacher (teacherB of Model157) for EVERY dev row.

The deployed GRUs (Model124) learned 0.5*y + 0.5*old-teacher.  For held fold k, teacherB_j (j not in {k,k+1};
same rows / seed / settings as models/model157/run_model157.py) is re-fitted and scores ALL rows of fold j
(the GRU trains on whole sequences) -> softB[k] exactly like Model124's `soft` array.
Output: models/model161/data/softB.npy  (5 x rows, float16; NaN for folds k and k+1)
Usage: python models/model161/soft_b.py
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

from tsauc import time_pair_weights, tsauc  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, protocol, layout, real, pool, old = m147.m111.load_everything()
    new = np.load(ROOT / "models/model157/data/real.npy", mmap_mode="r")
    union = protocol["contract"]["union_names"]
    soft = np.full((5, layout.row_count), np.nan, np.float16)
    for k in range(5):
        columns = np.asarray([union.index(n) for n in protocol["contract"]["names"][k]])
        for j in range(5):
            if j in (k, (k + 1) % 5):
                continue
            t0 = time.monotonic()
            pool_rows = np.flatnonzero(~np.isin(layout.row_fold, [k, (k + 1) % 5, j]))
            rows = np.sort(np.random.default_rng(7000 + 10 * k + j).choice(pool_rows, 450_000, replace=False))
            y, t = layout.target[rows], layout.time_online[rows]
            model = lgb.LGBMClassifier(objective="binary", random_state=31 + j, n_jobs=8, verbosity=-1,
                                       force_col_wise=True, **m147.m111.TEACHER_PARAMS)
            model.fit(np.hstack([real[rows][:, columns], np.asarray(old[rows]), np.asarray(new[rows])]), y,
                      sample_weight=time_pair_weights(y, t))
            target_rows = np.flatnonzero(layout.row_fold == j)
            prob = np.empty(len(target_rows))
            for a in range(0, len(target_rows), 250_000):
                r = target_rows[a:a + 250_000]
                prob[a:a + len(r)] = model.predict_proba(np.hstack([real[r][:, columns], np.asarray(old[r]), np.asarray(new[r])]))[:, 1]
            soft[k, target_rows] = 0.5 * layout.target[target_rows] + 0.5 * prob
            print(f"k={k} j={j}: teacherB TS-AUC {tsauc(layout.target[target_rows], prob, layout.time_online[target_rows]):.5f} "
                  f"on all {len(target_rows):,} rows [{time.monotonic() - t0:.0f}s]", flush=True)
    np.save(OUT / "softB.npy", soft)
    print("saved softB", soft.shape, flush=True)


if __name__ == "__main__":
    main()
