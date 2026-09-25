"""Model174: does a MIRRORED re-cut set earn a place among the pooled sets of the current student?

Model150 (Sub33-era recipe) found the mirrored re-cut set gave the best single tree student ever measured
here: +0.0088 dev, 5/5, against +0.0071/+0.0078/+0.0045 for the unflipped draws. The recipe then changed
(oracle-teacher soft targets on the synthetic rows, three re-cut sets pooled - Model163) and the mirrored
sets were never carried over: Model163 only ever pooled UNFLIPPED draws and found five no better than
three, which says nothing about a set that reverses direction instead of redrawing shifts.

This runs Model164's student (the dev-fold twin of the deployed recipe, so its numbers compare directly
with the SRP40 runs already in models/model164/runs) with the pooled set list overridden:
  SRP40  (reference, already run)   "", _s148, _s149
  FLIP3                             "", _s148, _f150     <- one unflipped draw swapped for a mirror
  FLIP4                             "", _s148, _s149, _f150
Model174 only overrides TAGS, the arm name and the output directory; every other line of the recipe is
Model164's, so the comparison changes one thing.
  python models/model174/run_model174.py FLIP3 0 1 2 3 4
Needs models/model174/teacher_flip.py to have built the mirrored set's teacher inputs first.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import models.model164.run_model164 as m164  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = {"FLIP3": ["", "_s148", "_f150"],
        "FLIP4": ["", "_s148", "_s149", "_f150"],
        "FLIP3b": ["", "_s148", "_f151"]}


def main():
    arm = sys.argv[1]
    folds = [int(a) for a in sys.argv[2:]] or [0]
    m164.TAGS = ARMS[arm]                      # the only change to the recipe
    m164.RUNS = HERE / "runs"                  # keep Model164's own runs untouched
    m164.RUNS.mkdir(parents=True, exist_ok=True)
    print(f"{arm}: pooled re-cut sets {m164.TAGS}", flush=True)
    _, protocol, layout, real, pool, _ = m164.m147.m111.load_everything()
    rec = m164.load_pooled()
    print(f"pooled re-cut rows {len(rec['target']):,}", flush=True)
    for fold in folds:
        t0 = time.monotonic()
        m164.student(protocol, layout, real, pool, rec, fold, arm=arm)
        ref = HERE.parent / f"model164/runs/SRP40_fold{fold}.npz"
        if ref.exists() and (m164.RUNS / f"{arm}_fold{fold}.npz").exists():
            from tsauc import tsauc
            a = np.load(m164.RUNS / f"{arm}_fold{fold}.npz"); b = np.load(ref)
            m = layout.row_fold == fold
            sa = tsauc(layout.target[m], a["prediction"], layout.time_online[m])
            sb = tsauc(layout.target[m], b["prediction"], layout.time_online[m])
            print(f"{arm} fold {fold}: {sa:.6f} vs SRP40 {sb:.6f} ({sa - sb:+.6f}) "
                  f"[{time.monotonic() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
