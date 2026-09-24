"""Model119: Model111 distillation with alpha = 0.75 (mostly hard labels).

Model113 (alpha 0.25) was slightly worse than Model111 (alpha 0.5); this is
the remaining side of the grid.  Teachers are Model111's cached cross-fitted
probabilities for the identical fit rows; everything else is Model111.

Usage: python models/model119/run_model119.py [fold ...]
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    runs = HERE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        source = m111.RUNS / f"fold{fold}_teacher.npz"
        if source.exists() and not (runs / source.name).exists():
            shutil.copy2(source, runs / source.name)
    m111.RUNS = runs
    m111.ALPHA = 0.75
    m111.TAG = "a075"
    m111.THREADS = 8
    m111.main()


if __name__ == "__main__":
    main()
