"""Model156: calibrated Bayesian break evidence, integrated uniformly over the unknown break position.

Differs from Model149 (Gaussian max-likelihood ratios, 30 columns, no gain) in the three ways the Discord
brief (models/DISCORD_NEXT_MODEL_BRIEF.md) says matter:
  * proper Bayes factors with AMPLITUDE PRIORS sized to real breaks (small), mixed within a channel;
  * a uniform prior over the break position tau (grid cells weighted by the number of tau they stand for:
    logsumexp(logBF(m_j) + log w_j) - log p), not equal weights over a geometric grid;
  * only three separate columns: scale, dependence, mean.
Input stream u_t: AR-whitened (fit on the history only, models/model57) and mapped through the HISTORY
empirical CDF to normal scores, so u ~ N(0,1) under "no break" whatever the marginal law.
  scale      x ~ N(0, v), v ~ InvGamma(a, b=a-1)                  a in A_SCALE
  mean       x ~ N(mu, 1), mu ~ N(0, s^2)                          s in S_MEAN
  dependence x_i | x_(i-1) ~ N(rho x_(i-1), 1 - rho^2)             rho in R_DEP
Every row reads only prefix sums up to its own step (leak checks: `check`).
Usage: python models/model156/bayes_evidence.py check | calibrate | real | syn | recuts
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
OUT = Path(__file__).resolve().parent / "data"
NAMES = ["be_scale", "be_dep", "be_mean"]
GRID = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768], np.int64)
A_SCALE = np.array([100.0, 30.0, 12.0, 6.0, 3.0])
S_MEAN = np.array([0.05, 0.12, 0.30])
R_DEP = np.array([0.06, -0.06, 0.12, -0.12, 0.20, -0.20, 0.35, -0.35])
HIST = 1000


@njit(cache=True)
def _ndtri(p):
    # Acklam's rational approximation (|rel err| < 1.2e-9)
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
def _lme(v, n):
    m = v[0]
    for i in range(1, n):
        if v[i] > m: m = v[i]
    s = 0.0
    for i in range(n): s += math.exp(v[i] - m)
    return m + math.log(s / n)


@njit(cache=True)
def _evidence(C1, C2, CX, CY, p, grid, a_s, s_m, r_d, out):
    """Uniform-in-tau evidence at p observed points; C* are prefix sums (index = number of points)."""
    nc = 0
    lo = 0
    cells_rep = np.empty(len(grid) + 1, np.int64); cells_w = np.empty(len(grid) + 1)
    for j in range(len(grid)):
        if lo >= p: break
        hi = min(grid[j], p)
        cells_rep[nc] = hi; cells_w[nc] = hi - lo; nc += 1; lo = grid[j]
    if lo < p:
        cells_rep[nc] = p; cells_w[nc] = p - lo; nc += 1
    tmp_s = np.empty(len(a_s)); tmp_m = np.empty(len(s_m)); tmp_d = np.empty(len(r_d))
    ls = np.empty(nc); lm = np.empty(nc); ld = np.empty(nc)
    for j in range(nc):
        m = cells_rep[j]
        S = C2[p] - C2[p - m]; T = C1[p] - C1[p - m]
        Sxy = CX[p] - CX[p - m]; Syy = CY[p] - CY[p - m]
        for k in range(len(a_s)):
            a = a_s[k]; b = a - 1.0
            tmp_s[k] = a * math.log(b) - math.lgamma(a) + math.lgamma(a + m / 2.0) - (a + m / 2.0) * math.log(b + S / 2.0) + S / 2.0
        for k in range(len(s_m)):
            s2 = s_m[k] * s_m[k]
            tmp_m[k] = -0.5 * math.log(1.0 + m * s2) + s2 * T * T / (2.0 * (1.0 + m * s2))
        for k in range(len(r_d)):
            r = r_d[k]; q = 1.0 - r * r
            tmp_d[k] = -0.5 * m * math.log(q) - (S - 2.0 * r * Sxy + r * r * Syy) / (2.0 * q) + S / 2.0
        lw = math.log(cells_w[j])
        ls[j] = _lme(tmp_s, len(a_s)) + lw; lm[j] = _lme(tmp_m, len(s_m)) + lw; ld[j] = _lme(tmp_d, len(r_d)) + lw
    lp = math.log(p)
    out[0] = _lme(ls, nc) + math.log(nc) - lp
    out[1] = _lme(ld, nc) + math.log(nc) - lp
    out[2] = _lme(lm, nc) + math.log(nc) - lp


@njit(cache=True)
def _batch(z, hs, ulast, grid, a_s, s_m, r_d):
    n = len(z); H = len(hs)
    X = np.zeros((n, 3), np.float32)
    C1 = np.zeros(n + 1); C2 = np.zeros(n + 1); CX = np.zeros(n + 1); CY = np.zeros(n + 1)
    prev = ulast; row = np.zeros(3)
    for t in range(n):
        u = _ndtri((np.searchsorted(hs, z[t]) + 0.5) / (H + 1.0))
        C1[t + 1] = C1[t] + u; C2[t + 1] = C2[t] + u * u
        CX[t + 1] = CX[t] + u * prev; CY[t + 1] = CY[t] + prev * prev
        prev = u
        _evidence(C1, C2, CX, CY, t + 1, grid, a_s, s_m, r_d, row)
        for j in range(3): X[t, j] = row[j]
    return X


def normal_scores(h, o):
    """History-only AR whitening + history-ECDF normal scores: (online z, sorted history tail, last history u)."""
    from models.model57.whitening import _fit_ar, _residuals
    h = np.asarray(h, np.float64); o = np.asarray(o, np.float64)
    order, intercept, phi = _fit_ar(h)
    hr, orr = _residuals(h, o, order, intercept, phi)
    hs = np.sort(hr[-HIST:])
    ulast = _ndtri((np.searchsorted(hs, hr[-1]) + 0.5) / (len(hs) + 1.0))
    return orr, hs, ulast


def series_features(h, o):
    z, hs, ulast = normal_scores(h, o)
    X = _batch(np.asarray(z, np.float64), hs, float(ulast), GRID, A_SCALE, S_MEAN, R_DEP)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def check():
    import pandas as pd
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).unique()[:8]
    rng = np.random.default_rng(0); t0 = time.time(); steps = 0
    for sid in ids:
        s = frame.loc[sid]; v = s["value"].to_numpy(np.float64); p = s["period"].to_numpy()
        h, o = v[p == 1], v[p == 2]
        full = series_features(h, o); steps += len(o); cut = len(o) // 2
        assert np.array_equal(series_features(h, o[:cut]), full[:cut]), "prefix invariance violated"
        o2 = o.copy(); o2[cut:] = o2[cut:] * 3 + rng.normal(size=len(o) - cut) * o.std()
        assert np.array_equal(series_features(h, o2)[:cut], full[:cut]), "future perturbation leaked"
        assert np.isfinite(full).all()
    print(f"leak checks passed on {len(ids)} series ({steps} steps, {time.time() - t0:.1f}s); "
          f"example last row {np.round(full[-1], 3)}", flush=True)


def calibrate():
    """Empirical break sizes on the normal-score stream (dev series with a break; descriptive only)."""
    import models.model111.run_model111 as m111
    _, protocol, layout, real, pool, _ = m111.load_everything()
    ls, dm, dr = [], [], []
    for p in sorted(pool):
        if layout.inner_series_fold[p] < 0:
            continue
        h, o = pool[p]; tau = int(layout.tau_index[p])
        if not (32 <= tau <= len(o) - 32):
            continue
        z, hs, _ = normal_scores(h, o)
        u = np.array([_ndtri((np.searchsorted(hs, x) + 0.5) / (len(hs) + 1.0)) for x in z])
        pre, post = u[:tau], u[tau:]
        ls.append(abs(np.log(post.std() / pre.std()))); dm.append(abs(post.mean() - pre.mean()))
        ac = lambda x: float(np.corrcoef(x[1:], x[:-1])[0, 1])  # noqa: E731
        dr.append(abs(ac(post) - ac(pre)))
    for name, v in (("|log sd ratio|", ls), ("|mean shift|", dm), ("|ac1 change|", dr)):
        print(f"{name}: n={len(v)} q25 {np.percentile(v, 25):.3f} median {np.median(v):.3f} p90 {np.percentile(v, 90):.3f}")
    print("(pre-segment noise alone gives nonzero values; compare with the prior grids in the module docstring)")


def build_real():
    import pandas as pd
    from models.model10.sources import load_development_layout
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy(np.float64); period = frame["period"].to_numpy()
    s0 = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); bounds = {int(ids[s]): (s, e) for s, e in zip(s0, np.r_[s0[1:], len(ids)])}
    pos = L.series_position; rs = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); re_ = np.r_[rs[1:], L.row_count]
    X = np.empty((L.row_count, 3), np.float32); t0 = time.time()
    for i, (a, b) in enumerate(zip(rs, re_)):
        s, e = bounds[int(L.series_ids[pos[a]])]; v, p = values[s:e], period[s:e]
        X[a:b] = series_features(v[p == 1], v[p == 2])
        if i % 2000 == 0: print(f"real: {i} series ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); np.save(OUT / "real.npy", X); print("saved real", X.shape, flush=True)
    from tsauc import tsauc
    for j, n in enumerate(NAMES):
        print(f"  single-column TS-AUC {n}: {tsauc(L.target, X[:, j], L.time_online):.4f}", flush=True)


def build_syn():
    import models.model111.run_model111 as m111
    from models.model20.run_model20 import synthetic_tasks
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for k in range(5):
        tasks, sy, st = synthetic_tasks(pool, protocol["plans"][k], layout, True)
        X = np.empty((len(sy), 3), np.float32); at = 0; t0 = time.time()
        for h, o, keep in tasks:
            f = series_features(h, o); X[at:at + len(keep)] = f[np.asarray(keep)]; at += len(keep)
        np.save(OUT / f"syn_fold{k}.npy", X); print(f"saved syn fold {k} {X.shape} ({time.time() - t0:.0f}s)", flush=True)


def build_recuts():
    import models.model111.run_model111 as m111
    _, protocol, layout, real, pool, _ = m111.load_everything()
    rec = np.load(ROOT / "models/model147/data/recuts.npz")
    parent, shift, t_on = rec["parent"], rec["shift"], rec["time_online"]
    X = np.empty((len(parent), 3), np.float32); t0 = time.time()
    starts = np.flatnonzero(np.r_[True, parent[1:] != parent[:-1]]); ends = np.r_[starts[1:], len(parent)]
    for a, b in zip(starts, ends):
        h, o = np.asarray(pool[int(parent[a])][0], np.float64), np.asarray(pool[int(parent[a])][1], np.float64)
        s = int(shift[a])
        X[a:b] = series_features(np.r_[h, o[:s]], o[s:])[t_on[a:b]]
    np.save(OUT / "recuts.npy", X); print(f"saved recuts {X.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"check": check, "calibrate": calibrate, "real": build_real, "syn": build_syn, "recuts": build_recuts}[sys.argv[1]]()
