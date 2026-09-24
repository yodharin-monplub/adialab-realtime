"""Bag of Model147 students: OOF logit mean of several arms -> runs/{TAG}_fold{k}.npz + dev gains.

Serving-equivalent: the bundle averages the logits of all fold voters of all arms, which for equal
numbers of voters per arm is the mean of the arms' bagged logits.
Usage: python models/model147/make_bag.py TAG ARM [ARM ...]      e.g.  BAG3 SRE40 SRE40b SRE40c
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main(tag, arms):
    L = load_development_layout(ROOT)
    table = {a: [] for a in arms}; table[tag] = []
    for k in range(5):
        m = L.row_fold == k; y, t = L.target[m], L.time_online[m]
        ref = tsauc(y, np.load(ROOT / f"models/model111/runs/a050_fold{k}.npz")["prediction"], t)
        zs = []
        for a in arms:
            d = np.load(RUNS / f"{a}_fold{k}.npz"); z = logit(d["prediction"]); zs.append(z)
            table[a].append(tsauc(y, z, t) - ref)
        z = np.mean(zs, axis=0)
        table[tag].append(tsauc(y, z, t) - ref)
        np.savez(RUNS / f"{tag}_fold{k}.npz", prediction=(1 / (1 + np.exp(-z))).astype(np.float32), held_rows=np.flatnonzero(m))
    for name, g in table.items():
        g = np.array(g)
        print(f"{name:10s} mean {g.mean():+.4f} s.e. {g.std(ddof=1) / 5 ** .5:.4f} wins {(g > 0).sum()}/5  folds " + " ".join(f"{v:+.4f}" for v in g))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
