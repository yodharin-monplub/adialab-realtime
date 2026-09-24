"""Why did the Model116 stacker lose 0.023? Controlled variants on held fold 0.

A: features [zc, zd, t] only, time-pair weights   -> must recover >= Submission30
B: full features, NO sample weights
C: full features, weights, min_child_samples 200
Prints TS-AUC vs Submission30 and top feature gains.
"""
from __future__ import annotations

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from models.model116.run_model116 import PARAMS, build_features  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

K = 0
TRAIN_ROWS = 600_000


def main():
    L = load_development_layout(ROOT)
    X, base = build_features(L)
    train = np.flatnonzero(L.row_fold != K)
    train = np.sort(np.random.default_rng(1).choice(train, TRAIN_ROWS, replace=False))
    held = L.row_fold == K
    y, t = L.target[train], L.time_online[train]
    w = time_pair_weights(y, t)
    ref = tsauc(L.target[held], base[held], L.time_online[held])
    print(f"Submission30 fold {K}: {ref:.6f}", flush=True)
    raw_cols = [0, 2 + len((2.0, 8.0, 32.0, 128.0)) + 3 + 4,  # zc raw (last of block c)
                2 + 2 * (len((2.0, 8.0, 32.0, 128.0)) + 3 + 5) - 1]  # zd raw (last of block d)
    variants = {
        "A_raw_scores_weighted": (X[:, raw_cols], w, PARAMS),
        "B_full_unweighted": (X, None, PARAMS),
        "C_full_weighted_small_leaves": (X, w, {**PARAMS, "min_child_samples": 200}),
    }
    for name, (F, sw, params) in variants.items():
        model = lgb.LGBMClassifier(objective="binary", random_state=0, n_jobs=2, verbosity=-1,
                                   force_col_wise=True, **params)
        model.fit(F[train], y, sample_weight=sw)
        p = model.predict_proba(F[held])[:, 1]
        s = tsauc(L.target[held], p, L.time_online[held])
        gain = model.booster_.feature_importance("gain")
        top = np.argsort(-gain)[:6]
        print(f"{name}: {s:.6f} ({s - ref:+.6f}); top gain cols {top.tolist()} "
              f"shares {np.round(gain[top] / gain.sum(), 3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
