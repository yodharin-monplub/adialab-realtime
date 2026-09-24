"""Model132: cross-family teacher average (tree-LLR teacher + bi-GRU teacher).

target = 0.5*y + 0.5 * mean(tree teacher prob, bi-GRU teacher prob) on real
rows; both teachers are fold-clean for held fold k and were built on the same
frozen fit rows (Model111 runs / Model131 runs).  Everything else is Model111.
Usage: python models/model132/run_model132.py 0 1
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import numpy as np  # noqa: E402

from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    runs = HERE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        out = runs / f"fold{fold}_teacher.npz"
        a = ROOT / f"models/model111/runs/fold{fold}_teacher.npz"
        b = ROOT / f"models/model131/runs/fold{fold}_teacher.npz"
        if out.exists() or not (a.exists() and b.exists()):
            continue
        da, db = np.load(a), np.load(b)
        assert np.array_equal(da["rows"], db["rows"])
        np.savez(out, rows=da["rows"], prob=(0.5 * da["prob"] + 0.5 * db["prob"]).astype(np.float32))
    folds = [int(x) for x in sys.argv[1:]] or [0, 1]
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    m111.RUNS = runs
    m111.ALPHA = 0.5
    m111.TAG = "a050mix"
    m111.THREADS = 8
    for fold in folds:
        pred = m111.run_fold(protocol, layout, real, pool, teacher, fold)
        held = layout.row_fold == fold
        ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
        print(f"fold {fold}: Model132 {tsauc(layout.target[held], pred, layout.time_online[held]):.6f} "
              f"vs Model111 {tsauc(layout.target[held], ref, layout.time_online[held]):.6f}", flush=True)


if __name__ == "__main__":
    main()
