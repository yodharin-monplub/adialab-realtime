"""Model120: age-softened labels — a hand-built teacher using only tau.

Hypothesis for why distillation helps: post-break rows at small age since
the break are nearly indistinguishable from pre-break rows, so their hard
label 1 is mostly noise.  Model120 replaces the learned teacher with
    target = y * (1 - 0.5 * exp(-age / 32)),   age = t - tau_index (>= 0)
(0.5 at the break, 0.82 at age 32, 0.93 at age 64, -> 1).  tau is used only
to build training targets; nothing privileged reaches inference.  Everything
else is Model111's frozen Model84 recipe (fit rows, synthetic rows with hard
labels, 40% synthetic mass, 317 columns, weights, expressive LightGBM).

Usage: python models/model120/run_model120.py [fold ...]
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SCALE = 32.0


def age_soft_targets(layout, real, teacher, columns, fold, fit):
    """Return a pseudo 'teacher' so that ALPHA*y + (1-ALPHA)*soft == y*g(age)."""
    y = layout.target[fit].astype(np.float32)
    tau = layout.tau_index[layout.series_position[fit]]
    age = np.maximum(layout.time_online[fit] - np.maximum(tau, 0), 0).astype(np.float32)
    g = 1.0 - 0.5 * np.exp(-age / SCALE)
    target = y * g
    # run_fold computes ALPHA*y + (1-ALPHA)*soft with ALPHA=0.5 -> soft = 2*target - y
    return (2.0 * target - y).astype(np.float32)


def main():
    m111.RUNS = HERE / "runs"
    m111.RUNS.mkdir(parents=True, exist_ok=True)
    m111.ALPHA = 0.5
    m111.TAG = "age32"
    m111.THREADS = 8
    m111.teacher_probabilities = age_soft_targets
    m111.main()


if __name__ == "__main__":
    main()
