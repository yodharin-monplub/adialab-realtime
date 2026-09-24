"""Model118: higher-capacity distilled student (Model111 with bigger trees).

Soft targets have far less label noise than 0/1 labels, so the student
should tolerate more capacity than Model84's expressive recipe (63 leaves,
1000 trees, lr 0.03).  Model118 uses 127 leaves, 2000 trees, lr 0.02,
min_child_samples 200; everything else (fit rows, synthetic mass, teacher
probabilities, 317 columns, weights) is Model111's.  Only the raw held-fold
component is reported; the top-level blend is evaluated with
models/model112/multi_blend.py.

Usage: python models/model118/run_model118.py [fold ...]
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, max_depth=-1,
           min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
           reg_alpha=0.2, reg_lambda=3.0, max_bin=127)


def main():
    runs = HERE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        source = m111.RUNS / f"fold{fold}_teacher.npz"
        if source.exists() and not (runs / source.name).exists():
            shutil.copy2(source, runs / source.name)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    _, protocol, layout, real, pool, teacher = m111.load_everything()  # verifies frozen Model84
    m111.RUNS = runs
    m111.ALPHA = 0.5
    m111.TAG = "a050big"
    m111.THREADS = 8
    m111.m84.CONFIG["arms"]["expressive"] = BIG  # after the frozen-protocol check
    for fold in folds:
        pred = m111.run_fold(protocol, layout, real, pool, teacher, fold)
        ref = np.load(m111.ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
        held = layout.row_fold == fold
        from tsauc import tsauc
        print(f"fold {fold}: Model118 {tsauc(layout.target[held], pred, layout.time_online[held]):.6f} "
              f"vs Model111 {tsauc(layout.target[held], ref, layout.time_online[held]):.6f}", flush=True)


if __name__ == "__main__":
    main()
