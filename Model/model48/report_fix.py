"""Replay frozen Model48 fits with validation appropriate for signed rank scores.

The frozen experiment reused Model18's probability-only checkpoint validator.
XGBRanker emits signed ranking margins, so that validator rejects valid saved
predictions. This wrapper changes validation only; all frozen data, fits,
checkpoints, selection weights, and reporting logic remain untouched.
"""
from __future__ import annotations

import fcntl
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model48 import run_model48 as experiment


def validate_rank_scores(saved, held):
    np.testing.assert_array_equal(saved["held_rows"], held)
    values = saved["prediction"]
    if values.shape != (len(held),) or values.dtype != np.float32:
        raise ValueError("Invalid signed rank-score dimensions or dtype")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 8):
        raise ValueError("Nonfinite or out-of-range signed rank scores")


def main():
    experiment.validate_predictions = validate_rank_scores
    with (experiment.HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, _, fingerprint = experiment.frozen_study()
        experiment.run(directory, fingerprint)


if __name__ == "__main__":
    main()
