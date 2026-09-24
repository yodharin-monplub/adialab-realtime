"""Grid of top-level blends: chain + sum_i w_i * z(logit(component_i)).

Weights in {0, 0.25, 0.5}; selection on pilot folds 0-1; confirmation folds 2-4
reported for the pilot-best combination and a few reference rows.
Usage: multi_blend.py comp1 comp2 ...   (names from COMPONENTS)
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent


def runs(n):
    return ROOT / json.loads((ROOT / f"models/model{n}/runs/latest.json").read_text())["directory"]


COMPONENTS = {
    "d111": lambda k: ROOT / f"models/model111/runs/a050_fold{k}.npz",
    "d113": lambda k: ROOT / f"models/model113/runs/a025_fold{k}.npz",
    "d114": lambda k: ROOT / f"models/model114/runs/a050_fold{k}.npz",
    "d115": lambda k: ROOT / f"models/model115/runs/a050r12_fold{k}.npz",
    "m62": lambda k: runs(62) / f"lightgbm_aligned_40_fold{k}.npz",
    "m63": lambda k: runs(63) / f"lightgbm_aligned_40_fold{k}.npz",
    "m64": lambda k: runs(64) / f"lightgbm_aligned_40_fold{k}.npz",
    "m84": lambda k: ROOT / f"models/model84/runs/e1e98a6e8af2df25/expressive_fold{k}.npz",
}
GRID = (0.0, 0.25, 0.5)


def logit_vector(layout, name):
    out = np.empty(layout.row_count)
    for k in range(5):
        p = np.clip(np.load(COMPONENTS[name](k))["prediction"].astype(np.float64), 1e-6, 1 - 1e-6)
        out[layout.row_fold == k] = np.log(p) - np.log1p(-p)
    return (out - out.mean()) / out.std()


def main(names):
    L = load_development_layout(ROOT)
    chain = np.load(HERE / "chain_m73_all.npy").astype(np.float64)
    chain = (chain - chain.mean()) / chain.std()
    comps = [logit_vector(L, n) for n in names]
    masks = [L.row_fold == k for k in range(5)]

    def scores(v):
        return np.array([tsauc(L.target[m], v[m], L.time_online[m]) for m in masks])

    base = scores(chain)
    results = []
    for weights in itertools.product(GRID, repeat=len(names)):
        v = chain + sum(w * c for w, c in zip(weights, comps))
        s = scores(v) - base
        results.append((weights, s))
        print(dict(zip(names, weights)), "pilot", round(s[:2].mean(), 5), "confirm", round(s[2:].mean(), 5),
              "all", round(s.mean(), 5), flush=True)
    best = max(results, key=lambda r: r[1][:2].mean())
    print("PILOT-BEST", dict(zip(names, best[0])), "fold gains", np.round(best[1], 5).tolist(),
          "confirm", round(best[1][2:].mean(), 5), "all", round(best[1].mean(), 5))
    top = sorted(results, key=lambda r: -r[1].mean())[:5]
    for w, s in top:
        print("ALL-TOP", dict(zip(names, w)), np.round(s, 5).tolist(), round(s.mean(), 5))


if __name__ == "__main__":
    main(sys.argv[1:])
