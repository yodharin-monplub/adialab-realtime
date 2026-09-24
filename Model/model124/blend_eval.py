"""Five-fold evaluation of a GRU prediction set on top of Submission30.

Usage: blend_eval.py "models/model124/remote_runs/{tag}fold{k}_pred.npz" [tag]
Reports raw GRU TS-AUC, Sub30, Sub30 + w*z(gru) for a weight grid, the
pilot-selected weight (folds 0-1) and its confirmation on folds 2-4.
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

WEIGHTS = (0.1, 0.15, 0.25, 0.35, 0.5)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main(pattern, tag=""):
    L = load_development_layout(ROOT)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    rows = []
    for k in range(5):
        if not all((ROOT / pat.format(tag=tag, k=k)).exists() for pat in pattern.split(",")):
            continue
        m = L.row_fold == k
        y, t = L.target[m], L.time_online[m]
        logits = []
        for pat in pattern.split(","):
            g = np.load(ROOT / pat.format(tag=tag, k=k))
            assert np.array_equal(g["held_rows"], np.flatnonzero(m))
            logits.append(lg(g["prediction"]))
        d111 = (lg(np.load(ROOT / f"models/model111/runs/a050_fold{k}.npz")["prediction"]) - c["mu_dist"]) / c["sd_dist"]
        zg = np.mean(logits, axis=0); zg = (zg - zg.mean()) / zg.std()
        sub30 = chain[m] + 0.5 * d111
        r = {"fold": k, "gru": tsauc(y, zg, t), "sub30": tsauc(y, sub30, t)}
        for w in WEIGHTS:
            r[w] = tsauc(y, sub30 + w * zg, t) - r["sub30"]
        rows.append(r)
        print(f"fold {k}: gru {r['gru']:.5f} sub30 {r['sub30']:.5f} gains " +
              " ".join(f"w{w}:{r[w]:+.5f}" for w in WEIGHTS), flush=True)
    pilot = [r for r in rows if r["fold"] in (0, 1)]
    confirm = [r for r in rows if r["fold"] in (2, 3, 4)]
    if pilot:
        w = max(WEIGHTS, key=lambda w: np.mean([r[w] for r in pilot]))
        print(f"pilot-selected w={w}: pilot gain {np.mean([r[w] for r in pilot]):+.5f}"
              + (f", confirmation gain {np.mean([r[w] for r in confirm]):+.5f} ({sum(r[w] > 0 for r in confirm)}/{len(confirm)} wins)" if confirm else ""))
    print("mean raw gru", round(float(np.mean([r["gru"] for r in rows])), 5),
          "mean gain by w", {w: round(float(np.mean([r[w] for r in rows])), 5) for w in WEIGHTS})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
