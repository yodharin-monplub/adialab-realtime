"""Model136: enforce/encourage monotone-in-time scores (user suggestion).

The label y_t is non-decreasing within a series (0...0 1...1).  The chain
already applies a windowed version (rolling max over 128 steps, 10% weight).
This probe applies the full constraint to Submission31's score:
  cummax mix :  s + w * (cummax(s) - s), w in {0.1..1}      (w=1 -> monotone)
  ratchet    :  envelope decaying toward s with factor d     (soft monotone)
on the development OOF (five folds) and on the sealed-holdout predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402


@njit(cache=True)
def cummax(x, pos):
    out = np.empty_like(x); m = -1e30
    for i in range(len(x)):
        if i == 0 or pos[i] != pos[i - 1]:
            m = x[i]
        elif x[i] > m:
            m = x[i]
        out[i] = m
    return out


@njit(cache=True)
def ratchet(x, pos, decay):
    out = np.empty_like(x); m = -1e30
    for i in range(len(x)):
        if i == 0 or pos[i] != pos[i - 1]:
            m = x[i]
        else:
            m = max(x[i], x[i] + decay * (m - x[i]))
        out[i] = m
    return out


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    L = load_development_layout(ROOT)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    g = json.loads((ROOT / "models/model124/constants_gru.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    d = np.empty(L.row_count); gru = np.zeros(L.row_count)
    for k in range(5):
        m = L.row_fold == k
        d[m] = lg(np.load(ROOT / f"models/model111/runs/a050_fold{k}.npz")["prediction"])
        for tag in ("", "h96l3_", "h128_"):
            gru[m] += lg(np.load(ROOT / f"models/model124/remote_runs/{tag}fold{k}_pred.npz")["prediction"]) / 3
    s31 = chain + 0.5 * (d - c["mu_dist"]) / c["sd_dist"] + 0.5 * (gru - g["mu_gru"]) / g["sd_gru"]
    pos = L.series_position.astype(np.int64)

    def folds(v):
        return np.array([tsauc(L.target[L.row_fold == k], v[L.row_fold == k], L.time_online[L.row_fold == k]) for k in range(5)])

    base = folds(s31)
    print("Submission31 dev folds", np.round(base, 5).tolist(), "mean", round(float(base.mean()), 6))
    cm = cummax(s31, pos)
    for w in (0.1, 0.25, 0.5, 0.75, 1.0):
        s = folds(s31 + w * (cm - s31))
        print(f"cummax mix w={w}: {np.round(s - base, 5).tolist()} mean {s.mean() - base.mean():+.5f}")
    for dec in (0.9, 0.97, 0.99):
        s = folds(ratchet(s31, pos, dec))
        print(f"ratchet decay={dec}: {np.round(s - base, 5).tolist()} mean {s.mean() - base.mean():+.5f}")
    r = np.load(ROOT / "models/model123/holdout_predictions.npz"); h = np.load(ROOT / "models/model124/holdout_predictions31.npz")
    y, t = r["target"], r["step"]; v = lg(h["final"])
    starts = np.flatnonzero(t == 0)
    hp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(t)])).astype(np.int64)
    hb = tsauc(y, v, t); hcm = cummax(v, hp)
    print(f"HOLDOUT Submission31 {hb:.5f}; cummax mix gains",
          {w: round(tsauc(y, v + w * (hcm - v), t) - hb, 5) for w in (0.1, 0.25, 0.5, 1.0)},
          "; ratchet gains", {dec: round(tsauc(y, ratchet(v, hp, dec), t) - hb, 5) for dec in (0.9, 0.97, 0.99)})


if __name__ == "__main__":
    main()
