"""Model130: synthetic-mass sweep under distilled targets.

The 40% synthetic cell mass (Model62) was tuned with hard labels. Under
distilled targets the real rows are less noisy, and the GRU/teacher show the
synthetic splices are far more detectable than real breaks, so the optimal
mass may differ.  Arms: synthetic_mass 0.2 and 0.6 (Model111 = 0.4), pilot
folds 0-1, everything else Model111.  Usage: python models/model130/run_model130.py 0 1
"""
from __future__ import annotations

from pathlib import Path
import shutil
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
        source = m111.RUNS / f"fold{fold}_teacher.npz"
        if source.exists() and not (runs / source.name).exists():
            shutil.copy2(source, runs / source.name)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1]
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    m111.RUNS = runs
    m111.ALPHA = 0.5
    m111.THREADS = 8
    for mass in (0.2, 0.6):
        m111.m84.CONFIG["synthetic_mass"] = mass   # after the frozen-protocol check in load_everything
        m111.TAG = f"a050m{int(mass * 100):02d}"
        for fold in folds:
            pred = m111.run_fold(protocol, layout, real, pool, teacher, fold)
            held = layout.row_fold == fold
            ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
            print(f"mass {mass} fold {fold}: {tsauc(layout.target[held], pred, layout.time_online[held]):.6f} "
                  f"vs Model111 (0.4) {tsauc(layout.target[held], ref, layout.time_online[held]):.6f}", flush=True)


if __name__ == "__main__":
    main()
