"""Model149: break EVIDENCE integrated over the unknown break position (second generation of Model144).

Model144 tests the series at the single most likely split.  Here, for every online step t and for each
of five channels of the AR-whitened, history-standardized series
    z (level), dz (first difference), |z| (scale), z_t * z_{t-1} (dependence), g = rank-Gauss(z) (shape-free)
the Gaussian mean+variance likelihood ratio LR_k of "break at k" vs "no break" is computed for ALL
candidate positions k (pre = history tail + o[0..k), post = o[k..t], >= 8 points), and summarised as
    ev_max    max_k 2 log LR_k                                (the Model144 statistic)
    ev_lme    log mean_k LR_k        = log marginal evidence under a uniform prior on the position
    ev_pos    posterior mean of k / p          ev_sd   posterior sd of k / p
    ev_r16 / ev_r64   posterior mass of "the break is within the last 16 / 64 steps"
= 6 x 5 = 30 causal columns.  Same leakage rules as Model144: tau is never read, row t uses only the
history and o[0..t], all constants come from the series' own history; `check` verifies prefix
invariance and that perturbing the future leaves earlier rows bit-identical.
Usage: python models/model149/evidence.py check | real | syn 0 1 2 3 4 | recuts [tag ...]
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
HIST = 1024
MINPOST = 8
CHANNELS = ("z", "dz", "abs", "lag", "rg")
STATS = ("max", "lme", "pos", "sd", "r16", "r64")
NAMES = [f"ev_{c}_{s}" for c in CHANNELS for s in STATS]
F = len(NAMES)


@njit(cache=True)
def _ndtri(p):
    """Acklam's rational approximation of the normal quantile (|error| < 1.2e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5; r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


@njit(cache=True)
def _channel(c1, c2, p, H, sh1, sh2, out, at):
    """Six evidence statistics of one channel for the prefix of length p (cumulative sums c1, c2)."""
    N = H + p
    tot1 = sh1 + c1[p]; tot2 = sh2 + c2[p]
    vall = max(tot2 / N - (tot1 / N) ** 2, 1e-12)
    n_k = p - MINPOST + 1
    best = -1e300
    half = np.empty(n_k)
    for k in range(n_k):
        n1 = H + k; n2 = p - k
        s1 = sh1 + c1[k]; q1 = sh2 + c2[k]
        s2 = c1[p] - c1[k]; q2 = c2[p] - c2[k]
        v1 = max(q1 / n1 - (s1 / n1) ** 2, 1e-12); v2 = max(q2 / n2 - (s2 / n2) ** 2, 1e-12)
        h = 0.5 * (N * math.log(vall) - n1 * math.log(v1) - n2 * math.log(v2))      # log LR_k
        half[k] = h
        if h > best: best = h
    tot = 0.0; m1 = 0.0; m2 = 0.0; r16 = 0.0; r64 = 0.0
    for k in range(n_k):
        w = math.exp(half[k] - best); tot += w
        x = k / p; m1 += w * x; m2 += w * x * x
        if k >= p - 16: r16 += w
        if k >= p - 64: r64 += w
    m1 /= tot; m2 /= tot
    out[at] = 2.0 * best
    out[at + 1] = best + math.log(tot / n_k)
    out[at + 2] = m1
    out[at + 3] = math.sqrt(max(m2 - m1 * m1, 0.0))
    out[at + 4] = r16 / tot
    out[at + 5] = r64 / tot


@njit(cache=True)
def _batch(z, zlast, hs, href, n):
    """All rows of one series; every row only reads prefix sums up to its own step."""
    out = np.zeros((n, F), np.float32); row = np.zeros(F)
    C1 = np.zeros((5, n + 1)); C2 = np.zeros((5, n + 1))
    H = len(hs); prev = zlast
    for t in range(n):
        zt = z[t]
        rank = np.searchsorted(hs, zt)                       # history ECDF -> normal score
        v = (zt, zt - prev, abs(zt), zt * prev, _ndtri((rank + 0.5) / (H + 1.0)))
        prev = zt
        for c in range(5):
            C1[c, t + 1] = C1[c, t] + v[c]; C2[c, t + 1] = C2[c, t] + v[c] * v[c]
        p = t + 1
        if p >= MINPOST:
            for c in range(5):
                _channel(C1[c], C2[c], p, int(href[c, 0]), href[c, 1], href[c, 2], row, 6 * c)
            for j in range(F): out[t, j] = row[j]
    return out


def series_features(h, o):
    from models.model57.whitening import _fit_ar, _residuals
    h = np.asarray(h, np.float64); o = np.asarray(o, np.float64)
    order, intercept, phi = _fit_ar(h)
    hr, orr = _residuals(h, o, order, intercept, phi)        # AR fitted on the history only
    hm, hsd = hr.mean(), hr.std() + 1e-12
    zh = (hr - hm) / hsd; z = (orr - hm) / hsd
    tail = zh[-HIST:]; hs = np.sort(tail); Hn = len(hs)
    rg = np.array([_ndtri((np.searchsorted(hs, x) + 0.5) / (Hn + 1.0)) for x in tail])
    chans = (tail, np.diff(zh)[-HIST:], np.abs(tail), (zh[1:] * zh[:-1])[-HIST:], rg)
    href = np.array([[len(c), c.sum(), (c * c).sum()] for c in chans], np.float64)
    X = _batch(z, float(zh[-1]), hs, href, len(z))
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def check():
    import pandas as pd
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).unique()[:6]
    rng = np.random.default_rng(0); t0 = time.time(); steps = 0; worst = 0.0
    for sid in ids:
        s = frame.loc[sid]; v = s["value"].to_numpy(np.float64); p = s["period"].to_numpy()
        h, o = v[p == 1], v[p == 2]
        full = series_features(h, o); steps += len(o); cut = len(o) // 2
        assert np.array_equal(series_features(h, o[:cut]), full[:cut]), "prefix invariance violated"
        o2 = o.copy(); o2[cut:] = o2[cut:] * 3 + rng.normal(size=len(o) - cut) * o.std()
        assert np.array_equal(series_features(h, o2)[:cut], full[:cut]), "future perturbation leaked"
        worst = max(worst, float(np.abs(full).max()))
    print(f"leak checks passed on {len(ids)} series ({steps} steps, {time.time() - t0:.1f}s); max |feature| {worst:.1f}")


def build_real():
    import pandas as pd
    from models.model10.sources import load_development_layout
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy(np.float64); period = frame["period"].to_numpy()
    s0 = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); bounds = {int(ids[s]): (s, e) for s, e in zip(s0, np.r_[s0[1:], len(ids)])}
    pos = L.series_position; rs = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); re_ = np.r_[rs[1:], L.row_count]
    X = np.empty((L.row_count, F), np.float32); t0 = time.time()
    for i, (a, b) in enumerate(zip(rs, re_)):
        s, e = bounds[int(L.series_ids[pos[a]])]; v, p = values[s:e], period[s:e]
        X[a:b] = series_features(v[p == 1], v[p == 2])
        if i % 1000 == 0: print(f"real: {i} series ({time.time() - t0:.0f}s)", flush=True)
    np.save(OUT / "real.npy", X); print("saved real", X.shape, flush=True)


def build_syn(folds):
    import models.model111.run_model111 as m111
    from models.model20.run_model20 import synthetic_tasks
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for k in folds:
        tasks, sy, st = synthetic_tasks(pool, protocol["plans"][k], layout, True)
        X = np.empty((len(sy), F), np.float32); at = 0; t0 = time.time()
        for h, o, keep in tasks:
            f = series_features(h, o); X[at:at + len(keep)] = f[np.asarray(keep)]; at += len(keep)
        np.save(OUT / f"syn_fold{k}.npy", X); print(f"saved syn fold {k} {X.shape} ({time.time() - t0:.0f}s)", flush=True)


def build_recuts(tags):
    """Same re-cuts as models/model147/build_recuts.py (same seeds => same cuts and kept rows)."""
    import models.model111.run_model111 as m111
    _, protocol, layout, real, pool, _ = m111.load_everything()
    seeds = {"": 147, "_s148": 148, "_s149": 149}
    for tag in tags:
        saved = np.load(ROOT / f"models/model147/data/recuts{tag}.npz")
        rng = np.random.default_rng(seeds[tag]); rows = []; parents = []; t0 = time.time()
        for p in sorted(pool):
            if layout.inner_series_fold[p] < 0:
                continue
            h, o = np.asarray(pool[p][0], np.float64), np.asarray(pool[p][1], np.float64)
            n, tau = len(o), int(layout.tau_index[p])
            if 0 <= tau < n:
                if tau < 24: continue
                s = int(rng.integers(8, tau - 8 + 1))
            else:
                if n < 32: continue
                s = int(rng.integers(8, n - 16 + 1))
            m = n - s
            keep = np.sort(rng.choice(m, min(50, m), replace=False))
            rows.append(series_features(np.r_[h, o[:s]], o[s:])[keep]); parents.append(np.full(len(keep), p, np.int32))
        X = np.concatenate(rows)
        assert np.array_equal(np.concatenate(parents), saved["parent"]), "re-cuts differ from Model147's"
        np.save(OUT / f"recuts{tag}.npy", X); print(f"saved recuts{tag} {X.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:] or ["check"]
    if "check" in args: check()
    if "real" in args: build_real()
    if "syn" in args: build_syn([int(a) for a in args if a.isdigit()])
    if "recuts" in args: build_recuts([a if a != "base" else "" for a in args[args.index("recuts") + 1:]] or [""])
