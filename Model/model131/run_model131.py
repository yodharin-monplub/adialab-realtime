"""Model131 student: Model111 recipe with bidirectional-GRU teacher targets.

Builds runs/fold{k}_teacher.npz (rows = Model111's frozen fit rows, prob =
bi-GRU teacher (k, j) probability for fit rows in fold j) from the downloaded
teacher_k{k}_j{j}.npz files, then runs Model111's student unchanged.
Usage: python models/model131/run_model131.py 0 1
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import numpy as np  # noqa: E402

from models.model20.run_model20 import fitting_rows  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
TEACHERS = HERE / "teachers"   # teacher_k{k}_j{j}.npz from the GPU box


def build_targets(layout, protocol, runs):
    for k in range(5):
        out = runs / f"fold{k}_teacher.npz"
        if out.exists():
            continue
        seed = protocol["contract"]["recipes"][k]["seed"]
        fit, _ = fitting_rows(layout, k, seed)
        prob = np.full(len(fit), np.nan, np.float32)
        report = {}
        for j in range(5):
            if j in (k, (k + 1) % 5):
                continue
            f = TEACHERS / f"teacher_k{k}_j{j}.npz"
            if not f.exists():
                return  # not all teachers present yet for this fold
            d = np.load(f)
            col = np.full(layout.row_count, np.nan, np.float32)
            col[d["rows_j"]] = d["prob_j"]
            mask = layout.row_fold[fit] == j
            prob[mask] = col[fit[mask]]
            report[j] = float(d["fold_j_tsauc"])
        assert np.isfinite(prob).all(), f"missing teacher probabilities for fold {k}"
        np.savez(out, rows=fit, prob=prob)
        print(f"fold {k}: bi-GRU teacher targets built; teacher fold-j TS-AUC {report}", flush=True)


def main():
    runs = HERE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1]
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    build_targets(layout, protocol, runs)
    m111.RUNS = runs
    m111.ALPHA = 0.5
    m111.TAG = "a050bigru"
    m111.THREADS = 8
    for fold in folds:
        pred = m111.run_fold(protocol, layout, real, pool, teacher, fold)
        held = layout.row_fold == fold
        ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
        print(f"fold {fold}: Model131 {tsauc(layout.target[held], pred, layout.time_online[held]):.6f} "
              f"vs Model111 {tsauc(layout.target[held], ref, layout.time_online[held]):.6f}", flush=True)


if __name__ == "__main__":
    main()
