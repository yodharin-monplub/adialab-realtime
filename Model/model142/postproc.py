"""Model142: causal post-processing of the Submission31 OOF score (no refitting).

score_t = z(chain) + .5 z(logit distilled) + .5 z(logit GRU) per row (dev OOF, Sub31 structure).
Each variant maps a series' score trajectory to a new trajectory using only steps <= t, then the
five held-fold TS-AUCs are compared with the untouched score.  TS-AUC ranks series at the same
step, so only series-specific temporal operations can change it.
Usage: python models/model142/postproc.py   -> stdout + score.txt
"""
from pathlib import Path
import json
import sys

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sub31_score(L):
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    d = np.empty(L.row_count); g = np.empty(L.row_count)
    for k in range(5):
        m = L.row_fold == k
        d[m] = lg(np.load(ROOT / f"models/model111/runs/a050_fold{k}.npz")["prediction"])
        g[m] = lg(np.load(ROOT / f"models/model141/runs/gru_fold{k}.npz")["prediction"])
    d = (d - c["mu_dist"]) / c["sd_dist"]; g = (g - g.mean()) / g.std()
    return chain + 0.5 * d + 0.5 * g


@njit(cache=True)
def ema_series(x, starts, ends, lam, out):
    for i in range(len(starts)):
        s = 0.0
        for t in range(starts[i], ends[i]):
            s = x[t] if t == starts[i] else lam * s + (1 - lam) * x[t]
            out[t] = s


@njit(cache=True)
def decaymax_series(x, starts, ends, delta, out):
    for i in range(len(starts)):
        m = -1e30
        for t in range(starts[i], ends[i]):
            m = max(x[t], m - delta); out[t] = m


@njit(cache=True)
def baseline_series(x, starts, ends, k, out):
    for i in range(len(starts)):
        s = 0.0; n = 0
        for t in range(starts[i], ends[i]):
            if n < k:
                s += x[t]; n += 1
            out[t] = s / n


@njit(cache=True)
def pastrank_series(x, starts, ends, out):
    """fraction of the series' earlier steps with a lower score (causal own-history rank)."""
    for i in range(len(starts)):
        for t in range(starts[i], ends[i]):
            n = t - starts[i]
            if n == 0:
                out[t] = 0.5; continue
            c = 0
            for u in range(starts[i], t):
                if x[u] < x[t]: c += 1
            out[t] = c / n


def main():
    L = load_development_layout(ROOT)
    x = sub31_score(L)
    pos = L.series_position
    starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]).astype(np.int64); ends = np.r_[starts[1:], L.row_count].astype(np.int64)
    masks = [L.row_fold == k for k in range(5)]

    def scores(v):
        return np.array([tsauc(L.target[m], v[m], L.time_online[m]) for m in masks])

    base = scores(x)
    lines = [f"untouched Sub31 OOF: folds {np.round(base, 5).tolist()} mean {base.mean():.5f}"]
    print(lines[-1], flush=True)
    variants = {}
    tmp = np.empty_like(x)
    for h in (2, 4, 8, 16, 32):
        lam = 0.5 ** (1 / h); ema_series(x, starts, ends, lam, tmp); e = tmp.copy()
        for w in (0.3, 0.6, 1.0):
            variants[f"ema h={h} w={w}"] = (1 - w) * x + w * e
        for gm in (0.25, 0.5, 1.0):
            variants[f"momentum h={h} g={gm}"] = x + gm * (x - e)
    for delta in (0.01, 0.03, 0.1, 0.3):
        decaymax_series(x, starts, ends, delta, tmp); dm = tmp.copy()
        for w in (0.1, 0.3, 0.6):
            variants[f"decaymax d={delta} w={w}"] = (1 - w) * x + w * dm
    for k in (8, 16, 32, 64):
        baseline_series(x, starts, ends, k, tmp); b = tmp.copy()
        for a in (0.25, 0.5, 1.0):
            variants[f"baseline k={k} a={a}"] = x - a * b
    pastrank_series(x, starts, ends, tmp); pr = tmp.copy()
    for w in (0.25, 0.5, 1.0):
        variants[f"pastrank w={w}"] = x + w * pr
    results = []
    for name, v in variants.items():
        s = scores(v) - base
        results.append((name, s))
        print(f"{name:28s} gain folds {np.round(s, 5).tolist()} mean {s.mean():+.5f} wins {(s > 0).sum()}/5", flush=True)
    results.sort(key=lambda r: -r[1].mean())
    lines.append("top 10 by mean gain (5 folds):")
    for name, s in results[:10]:
        lines.append(f"  {name:28s} {s.mean():+.5f}  s.e. {s.std(ddof=1) / np.sqrt(5):.5f}  wins {(s > 0).sum()}/5  " + " ".join(f"{g:+.4f}" for g in s))
    print("\n".join(lines[1:]))
    (HERE / "score.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
