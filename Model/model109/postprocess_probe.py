"""Probe causal per-series smoothing of Model109 student predictions (pilot folds)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


@njit(cache=True)
def ewma(x, pos, alpha):
    out = np.empty_like(x)
    for i in range(len(x)):
        if i == 0 or pos[i] != pos[i - 1]:
            s = x[i]
        else:
            s = alpha * x[i] + (1 - alpha) * s
        out[i] = s
    return out


@njit(cache=True)
def running_max(x, pos, window):
    out = np.empty_like(x)
    start = 0
    for i in range(len(x)):
        if i == 0 or pos[i] != pos[i - 1]:
            start = i
        lo = max(start, i - window + 1)
        m = x[lo]
        for j in range(lo + 1, i + 1):
            if x[j] > m:
                m = x[j]
        out[i] = m
    return out


def main(run_dir: Path, arms):
    L = load_development_layout(ROOT)
    for k in (0, 1):
        held = np.flatnonzero(L.row_fold == k)
        y, t, pos = L.target[held], L.time_online[held], L.series_position[held]
        for arm in arms:
            f = run_dir / f"fold{k}_{arm}.npy"
            if not f.exists():
                continue
            z = logit(np.load(f))
            res = {"raw": tsauc(y, z, t)}
            for a in (0.5, 0.3, 0.15):
                res[f"ewma{a}"] = tsauc(y, ewma(z, pos, a), t)
            rm = running_max(z, pos, 128)
            for w in (0.1, 0.25, 0.5):
                res[f"max128_{w}"] = tsauc(y, z + w * (rm - z), t)
            e = ewma(z, pos, 0.3)
            rme = running_max(e, pos, 128)
            res["ewma.3+max.25"] = tsauc(y, e + 0.25 * (rme - e), t)
            print(k, arm, {a: round(b, 5) for a, b in res.items()}, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
