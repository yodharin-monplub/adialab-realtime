"""Model135 trainer: Model134's arms on data/inject.npz (parametric injections).

Usage: python models/model135/run_model135.py 0 1
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model134.run_model134 as base  # noqa: E402  (installs Linux stubs via model111)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    base.RUNS = HERE / "runs"
    base.RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1]
    _, protocol, layout, real, pool, _ = base.m111.load_everything()
    boot = np.load(HERE / "data" / "inject.npz")
    boot = {k: boot[k] for k in boot.files}
    for arm in ("replace", "add"):
        for fold in folds:
            base.run_fold(protocol, layout, real, pool, boot, fold, arm)


if __name__ == "__main__":
    main()
