"""Replay the frozen Model44 report with its Model26 checkpoint path supplied.

The original report mistakenly passes ``None`` to Model26's checkpoint loader.
Keep the frozen training and report files unchanged; repair only that call at
runtime so the completed fit checkpoints remain tied to their original hash.
"""
from __future__ import annotations

import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model26 import run_model26 as incumbent
from models.model44.run_model44 import HERE, frozen_study


def main() -> None:
    frozen_study()
    directory, _, fingerprint = incumbent.frozen_study()
    original = incumbent.prediction_checkpoint

    def corrected_checkpoint(given_directory, arm, fold, given_fingerprint, held):
        if given_directory is None and given_fingerprint is None:
            if arm != "depth_mix":
                raise ValueError("Unexpected Model44 checkpoint arm")
            return original(directory, arm, fold, fingerprint, held)
        return original(given_directory, arm, fold, given_fingerprint, held)

    incumbent.prediction_checkpoint = corrected_checkpoint
    from models.model44.report import report

    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report()


if __name__ == "__main__":
    main()
