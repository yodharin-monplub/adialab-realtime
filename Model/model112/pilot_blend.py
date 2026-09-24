"""Pilot-fold evaluation of a candidate distilled voter in the Submission30 blend.

For every fold whose file exists:
  chain            Submission28 chain (pooled z-score)
  sub30            chain + 0.5 * z(Model111)            (deployed)
  cand@w           chain + w * z(candidate)             (candidate alone)
  sub30+cand@w     chain + 0.5*z(Model111) + w*z(cand)  (added on top)
Candidate/Model111 z-scores use the pooled Model111 constants for Model111 and
per-fold standardization for the candidate (approximation for pilot use).
Usage: pilot_blend.py "models/model117/runs/a050s3_fold{k}.npz"
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
WEIGHTS = (0.25, 0.5, 0.75)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main(pattern):
    L = load_development_layout(ROOT)
    c = json.loads((HERE / "constants.json").read_text())
    chain = (np.load(HERE / "chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    rows = []
    for k in range(5):
        f = ROOT / pattern.format(k=k)
        if not f.exists():
            continue
        m = L.row_fold == k
        y, t = L.target[m], L.time_online[m]
        d111 = (lg(np.load(ROOT / f"models/model111/runs/a050_fold{k}.npz")["prediction"]) - c["mu_dist"]) / c["sd_dist"]
        cand = lg(np.load(f)["prediction"])
        cand = (cand - cand.mean()) / cand.std()
        r = {"fold": k, "chain": tsauc(y, chain[m], t), "sub30": tsauc(y, chain[m] + 0.5 * d111, t)}
        for w in WEIGHTS:
            r[f"cand@{w}"] = tsauc(y, chain[m] + w * cand, t)
            r[f"sub30+cand@{w}"] = tsauc(y, chain[m] + 0.5 * d111 + w * cand, t)
        rows.append(r)
        print({a: (round(b, 5) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    if rows:
        keys = [k for k in rows[0] if k != "fold"]
        print("mean", {k: round(float(np.mean([r[k] for r in rows])), 5) for k in keys})


if __name__ == "__main__":
    main(sys.argv[1])
