"""Model121: age-softened hard label combined with the learned teacher.

    target = 0.5 * y * g(age) + 0.5 * teacher,   g(age) = 1 - 0.5*exp(-age/32)
Model120 tests g(age) alone; Model121 tests whether the two sources of
softening are complementary.  Teacher probabilities are Model111's cached
cross-fitted values (identical fit rows); everything else is Model111.

Usage: python models/model121/run_model121.py [fold ...]
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
SCALE = 32.0
_cached = m111.teacher_probabilities


def combined_targets(layout, real, teacher, columns, fold, fit):
    prob = _cached(layout, real, teacher, columns, fold, fit)  # loads the copied cache
    y = layout.target[fit].astype(np.float32)
    tau = layout.tau_index[layout.series_position[fit]]
    age = np.maximum(layout.time_online[fit] - np.maximum(tau, 0), 0).astype(np.float32)
    g = 1.0 - 0.5 * np.exp(-age / SCALE)
    target = 0.5 * y * g + 0.5 * prob
    return (2.0 * target - y).astype(np.float32)  # run_fold forms 0.5*y + 0.5*soft


def main():
    runs = HERE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        source = m111.RUNS / f"fold{fold}_teacher.npz"
        if source.exists() and not (runs / source.name).exists():
            shutil.copy2(source, runs / source.name)
    m111.RUNS = runs
    m111.ALPHA = 0.5
    m111.TAG = "age32t"
    m111.THREADS = 8
    m111.teacher_probabilities = combined_targets
    m111.main()


if __name__ == "__main__":
    main()
