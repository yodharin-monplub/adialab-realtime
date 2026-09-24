"""Model113: second oracle-distilled Model84-recipe voter (alpha = 0.25).

Identical to Model111 except the real-row target mix (0.25*y + 0.75*teacher).
Model111's cached teacher probabilities for the same frozen fit rows are
reused (copied into runs/), so no teacher is refit.  The purpose is a
variance-reducing second distilled voter for the Model112 top-level blend.

Usage: python models/model113/run_model113.py [fold ...]
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
    m111.ALPHA = 0.25
    m111.TAG = "a025"
    m111.THREADS = 6
    m111.main()


if __name__ == "__main__":
    main()
