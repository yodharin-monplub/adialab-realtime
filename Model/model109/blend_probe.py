"""Blend student predictions into Model84's final OOF score on available folds.

candidate = z(Model84) + w * z(logit(student)), z = global standardization over
the held fold.  Usage: blend_probe.py RUN_DIR ARM [ARM ...]
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def z(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std()


def main(run_dir: Path, arms):
    L = load_development_layout(ROOT)
    m84 = np.load(ROOT / "models/model109/cache/model84_final_oof.npy")
    for arm in arms:
        gains = {w: [] for w in WEIGHTS}
        for k in range(5):
            f = run_dir / f"fold{k}_{arm}.npy"
            if not f.exists():
                continue
            held = L.row_fold == k
            y, t = L.target[held], L.time_online[held]
            a, b = z(m84[held]), z(logit(np.load(f)))
            base = tsauc(y, a, t)
            row = []
            for w in WEIGHTS:
                s = tsauc(y, a + w * b, t)
                gains[w].append(s - base)
                row.append(f"{w}:{s:.5f}")
            print(f"{arm} fold{k} base {base:.5f} | " + " ".join(row), flush=True)
        print(arm, "mean gain", {w: round(float(np.mean(g)), 5) for w, g in gains.items() if g})


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
