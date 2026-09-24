"""Model144 (user's idea, 2026-09-20): two-sample tests between PRE and POST at an ESTIMATED split.

At online step t (prefix o[0..t]) the break time is unknown, so:
  1. scan every candidate split k (post = o[k..t], at least MINPOST points; pre = history tail + o[0..k))
     with the Gaussian mean+variance likelihood ratio  LLR(k) = N log v_all - n1 log v_pre - n2 log v_post
     -> k_hat = argmax, plus how sharp / how stable that estimate is;
  2. run a battery of two-sample tests between the clean reference (history tail; pooled pre for the
     moment tests) and the estimated post segment o[k_hat..t] (last 512 points at most):
     Welch t, log variance ratio, KS, Kuiper, Wasserstein-1, Mann-Whitney z, mean-abs-deviation ratio,
     lag-1 autocorrelation, kurtosis, roughness.
  The same scan is repeated on the first-differenced series (stage-2 winner) with a smaller battery.

LEAKAGE RULES: tau is never read here; row t uses only the history and o[0..t]; every constant comes
from the series' own history.  `row_features` computes ONE step from the prefix (+ a tiny carried
state for the stability counters); the batch builder is a loop over it, so serving = the same code.
check() verifies prefix-invariance (features of a truncated series == features of the full series).

Usage: python models/model144/split_tests.py check | real | syn 0 1 2 3 4
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
MAXPOST = 512
NAMES = ["sp_llr_max", "sp_llr_k0", "sp_llr_sharp", "sp_khat_rel", "sp_log_post", "sp_age",
         "sp_welch_t", "sp_abs_welch", "sp_logvr", "sp_abs_logvr", "sp_ks", "sp_kuiper", "sp_wasser",
         "sp_mwu_absz", "sp_mad_ratio", "sp_ac1_diff", "sp_kurt_diff", "sp_rough_ratio",
         "spd_llr_max", "spd_llr_k0", "spd_llr_sharp", "spd_khat_rel", "spd_log_post", "spd_age",
         "spd_logvr", "spd_abs_welch", "sp_khat_agree"]
F = len(NAMES)


@njit(cache=True)
def _scan(c1, c2, p, H, sh1, sh2):
    """Gaussian mean+variance LLR over splits k = 0..p-MINPOST. Returns max, at k=0, mean, argmax."""
    N = H + p
    tot1 = sh1 + c1[p]; tot2 = sh2 + c2[p]
    vall = max(tot2 / N - (tot1 / N) ** 2, 1e-12)
    best = -1e30; bk = 0; acc = 0.0; cnt = 0; at0 = 0.0
    for k in range(0, p - MINPOST + 1):
        n1 = H + k; n2 = p - k
        s1 = sh1 + c1[k]; q1 = sh2 + c2[k]
        s2 = c1[p] - c1[k]; q2 = c2[p] - c2[k]
        v1 = max(q1 / n1 - (s1 / n1) ** 2, 1e-12); v2 = max(q2 / n2 - (s2 / n2) ** 2, 1e-12)
        llr = N * math.log(vall) - n1 * math.log(v1) - n2 * math.log(v2)
        if k == 0: at0 = llr
        acc += llr; cnt += 1
        if llr > best: best = llr; bk = k
    return best, at0, acc / cnt, bk


@njit(cache=True)
def _ac1(x):
    n = len(x); m = x.mean(); d = 0.0; s = 0.0
    for i in range(n):
        d += (x[i] - m) ** 2
        if i > 0: s += (x[i] - m) * (x[i - 1] - m)
    return s / d if d > 0 else 0.0


@njit(cache=True)
def _kurt(x):
    n = len(x); m = x.mean(); s2 = 0.0; s4 = 0.0
    for i in range(n):
        d = x[i] - m; s2 += d * d; s4 += d ** 4
    return n * s4 / (s2 * s2) - 3.0 if s2 > 0 else 0.0


@njit(cache=True)
def _rough(x):
    s = 0.0
    for i in range(1, len(x)): s += abs(x[i] - x[i - 1])
    return s / max(len(x) - 1, 1)


@njit(cache=True)
def row_features(z, dz, c1, c2, d1, d2, p, hs, href, state, out):
    """Features of the online prefix of length p (step t = p-1). z/dz and their cumulative sums are
    only read up to index p.  href = history constants, state = carried [prev_khat, age, prev_khat_d, age_d]."""
    H = int(href[0]); sh1 = href[1]; sh2 = href[2]; hbf = href[3]; hac = href[4]; hku = href[5]; hro = href[6]
    Hd = int(href[7]); sd1 = href[8]; sd2 = href[9]
    for j in range(F): out[j] = 0.0
    if p < MINPOST:
        return
    best, at0, mean_llr, k = _scan(c1, c2, p, H, sh1, sh2)
    state[1] = state[1] + 1 if abs(k - state[0]) <= 2 else 0.0
    state[0] = k
    out[0] = best; out[1] = at0; out[2] = best - mean_llr; out[3] = k / p; out[4] = math.log1p(p - k); out[5] = math.log1p(state[1])
    start = max(k, p - MAXPOST); m = p - start
    n1 = H + k; s1 = sh1 + c1[k]; q1 = sh2 + c2[k]
    m1 = s1 / n1; v1 = max(q1 / n1 - m1 * m1, 1e-12)
    s2 = c1[p] - c1[start]; q2 = c2[p] - c2[start]
    m2 = s2 / m; v2 = max(q2 / m - m2 * m2, 1e-12)
    t = (m2 - m1) / math.sqrt(v1 / n1 + v2 / m)
    out[6] = t; out[7] = abs(t); out[8] = math.log(v2 / v1); out[9] = abs(out[8])
    w = z[start:p]; ws = np.sort(w); n = len(hs)
    i = 0; j = 0; dplus = 0.0; dminus = 0.0; U = 0.0
    while i < n or j < m:                      # merge: history first on ties
        if j >= m or (i < n and hs[i] <= ws[j]):
            i += 1
        else:
            U += i; j += 1                      # history values <= this post value
        fh = i / n; fw = j / m
        if fw - fh > dplus: dplus = fw - fh
        if fh - fw > dminus: dminus = fh - fw
    scale = math.sqrt(n * m / (n + m))
    out[10] = max(dplus, dminus) * scale; out[11] = (dplus + dminus) * scale
    wd = 0.0
    for q in range(m):
        wd += abs(ws[q] - hs[min(n - 1, int((q + 0.5) / m * n))])
    out[12] = wd / m
    out[13] = abs((U - n * m / 2.0) / math.sqrt(n * m * (n + m + 1) / 12.0))
    med = ws[m // 2] if m % 2 == 1 else 0.5 * (ws[m // 2 - 1] + ws[m // 2])
    mad = 0.0
    for q in range(m): mad += abs(w[q] - med)
    out[14] = math.log((mad / m + 1e-12) / hbf)
    out[15] = _ac1(w) - hac; out[16] = _kurt(w) - hku; out[17] = math.log((_rough(w) + 1e-12) / hro)
    # differenced series
    bestd, at0d, meand, kd = _scan(d1, d2, p, Hd, sd1, sd2)
    state[3] = state[3] + 1 if abs(kd - state[2]) <= 2 else 0.0
    state[2] = kd
    out[18] = bestd; out[19] = at0d; out[20] = bestd - meand; out[21] = kd / p; out[22] = math.log1p(p - kd); out[23] = math.log1p(state[3])
    startd = max(kd, p - MAXPOST); md = p - startd
    n1d = Hd + kd; a1 = sd1 + d1[kd]; b1 = sd2 + d2[kd]
    mu1 = a1 / n1d; var1 = max(b1 / n1d - mu1 * mu1, 1e-12)
    a2 = d1[p] - d1[startd]; b2 = d2[p] - d2[startd]
    mu2 = a2 / md; var2 = max(b2 / md - mu2 * mu2, 1e-12)
    out[24] = math.log(var2 / var1); out[25] = abs((mu2 - mu1) / math.sqrt(var1 / n1d + var2 / md))
    out[26] = abs(k - kd) / p


def history_reference(h):
    """Everything taken from the history (known clean): z-scaling, sorted tail, reference constants."""
    h = np.asarray(h, np.float64)
    hm, hsd = h.mean(), h.std() + 1e-12
    zh = ((h - hm) / hsd)[-HIST:]
    dh = np.diff((h - hm) / hsd)[-HIST:]
    med = np.median(zh)
    href = np.array([len(zh), zh.sum(), (zh * zh).sum(), np.mean(np.abs(zh - med)) + 1e-12, _ac1(zh), _kurt(zh),
                     _rough(zh) + 1e-12, len(dh), dh.sum(), (dh * dh).sum()], np.float64)
    return hm, hsd, np.sort(zh), href, ((h[-1] - hm) / hsd)


@njit(cache=True)
def _batch(z, dz, hs, href, n):
    out = np.zeros((n, F), np.float32); row = np.zeros(F); state = np.array([-10.0, 0.0, -10.0, 0.0])
    c1 = np.zeros(n + 1); c2 = np.zeros(n + 1); d1 = np.zeros(n + 1); d2 = np.zeros(n + 1)
    for t in range(n):                         # cumulative sums are extended one step at a time
        c1[t + 1] = c1[t] + z[t]; c2[t + 1] = c2[t] + z[t] * z[t]
        d1[t + 1] = d1[t] + dz[t]; d2[t + 1] = d2[t] + dz[t] * dz[t]
        row_features(z, dz, c1, c2, d1, d2, t + 1, hs, href, state, row)
        for j in range(F): out[t, j] = row[j]
    return out


WHITEN = __import__("os").environ.get("SP_WHITEN", "0") == "1"   # variant: run everything on AR innovations
if WHITEN:
    OUT = HERE / "data_white"


def whiten(h, o):
    """AR(p) fitted on the HISTORY only (BIC over p in 0,1,2,4,8,12 - Model57/73 code); returns the
    one-step-ahead residuals of the history and of the online part. Causal: residual t needs o[<=t]."""
    from models.model57.whitening import _fit_ar, _residuals
    order, intercept, phi = _fit_ar(h)
    return _residuals(h, o, order, intercept, phi)


def series_features(h, o):
    if WHITEN:
        h, o = whiten(np.asarray(h, np.float64), np.asarray(o, np.float64))
    hm, hsd, hs, href, last = history_reference(h)
    z = (np.asarray(o, np.float64) - hm) / hsd
    dz = np.diff(np.concatenate(([last], z)))
    X = _batch(z, dz, hs, href, len(z))
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


class StreamingSplitTests:
    """Serving path: one call per online value, same row_features() as the batch builder."""

    def __init__(self, history):
        h = np.asarray(history, np.float64)
        self.order = 0
        if WHITEN:
            from models.model57.whitening import _fit_ar, _residuals
            self.order, self.intercept, self.phi = _fit_ar(h)
            self.recent = list(h[-self.order:]) if self.order else []
            h = _residuals(h, h[:0], self.order, self.intercept, self.phi)[0]
        self.hm, self.hsd, self.hs, self.href, self.prev = history_reference(h)
        cap = 4096
        self.z = np.zeros(cap); self.dz = np.zeros(cap)
        self.c1 = np.zeros(cap + 1); self.c2 = np.zeros(cap + 1); self.d1 = np.zeros(cap + 1); self.d2 = np.zeros(cap + 1)
        self.state = np.array([-10.0, 0.0, -10.0, 0.0]); self.row = np.zeros(F); self.p = 0

    def _grow(self):
        for name in ("z", "dz", "c1", "c2", "d1", "d2"):
            old = getattr(self, name); new = np.zeros(2 * len(old)); new[:len(old)] = old; setattr(self, name, new)

    def step(self, value):
        x = float(value)
        if WHITEN:
            r = x - self.intercept
            if self.order:
                r -= float(np.asarray(self.recent[::-1]) @ self.phi)
                self.recent = self.recent[1:] + [x]
            x = r
        if self.p + 1 >= len(self.z):
            self._grow()
        t = self.p; zt = (x - self.hm) / self.hsd
        self.z[t] = zt; self.dz[t] = zt - self.prev; self.prev = zt
        self.c1[t + 1] = self.c1[t] + zt; self.c2[t + 1] = self.c2[t] + zt * zt
        self.d1[t + 1] = self.d1[t] + self.dz[t]; self.d2[t + 1] = self.d2[t] + self.dz[t] * self.dz[t]
        self.p = t + 1
        row_features(self.z, self.dz, self.c1, self.c2, self.d1, self.d2, self.p, self.hs, self.href, self.state, self.row)
        return np.nan_to_num(self.row, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def parity():
    """Streaming (serving) rows must equal the batch (training) rows."""
    import pandas as pd
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).unique()[:8]
    worst = 0.0; steps = 0
    for sid in ids:
        s = frame.loc[sid]; v = s["value"].to_numpy(np.float64); p = s["period"].to_numpy()
        h, o = v[p == 1], v[p == 2]
        ref = series_features(h, o)
        st = StreamingSplitTests(h.astype(np.float32))      # serving passes float32 values
        got = np.stack([st.step(np.float32(x)) for x in o])
        scale = np.abs(ref).max(axis=0) + 1e-6
        worst = max(worst, float((np.abs(got - ref) / scale).max())); steps += len(o)
    print(f"streaming vs batch on {len(ids)} series / {steps} steps (whiten={WHITEN}): worst relative difference {worst:.2e}")


def check():
    """Leak checks: (1) prefix invariance, (2) perturbing the future leaves earlier rows bit-identical."""
    import pandas as pd
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).unique()[:6]
    rng = np.random.default_rng(0); worst = 0.0; t0 = time.time(); steps = 0
    for sid in ids:
        s = frame.loc[sid]; v = s["value"].to_numpy(np.float64); p = s["period"].to_numpy()
        h, o = v[p == 1], v[p == 2]
        full = series_features(h, o); steps += len(o)
        cut = len(o) // 2
        assert np.array_equal(series_features(h, o[:cut]), full[:cut]), "prefix invariance violated"
        o2 = o.copy(); o2[cut:] = o2[cut:] * 3 + rng.normal(size=len(o) - cut) * o.std()
        assert np.array_equal(series_features(h, o2)[:cut], full[:cut]), "future perturbation leaked"
        assert not np.array_equal(series_features(h, o2)[cut:], full[cut:])
        worst = max(worst, float(np.abs(full).max()))
    print(f"leak checks passed on {len(ids)} series ({steps} steps, {time.time() - t0:.1f}s incl. compile); max |feature| {worst:.1f}")


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
        if i % 500 == 0: print(f"real: {i} series ({time.time() - t0:.0f}s)", flush=True)
    np.save(OUT / "real.npy", X); print("saved real", X.shape, flush=True)


def build_syn(folds):
    import models.model111.run_model111 as m111
    from models.model20.run_model20 import synthetic_tasks
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for k in folds:
        tasks, sy, st = synthetic_tasks(pool, protocol["plans"][k], layout, True)
        X = np.empty((len(sy), F), np.float32); at = 0; t0 = time.time()
        for h, o, keep in tasks:
            f = series_features(np.asarray(h, np.float64), np.asarray(o, np.float64))
            X[at:at + len(keep)] = f[np.asarray(keep)]; at += len(keep)
        np.save(OUT / f"syn_fold{k}.npy", X); print(f"saved syn fold {k} {X.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:] or ["check"]
    if "check" in args: check()
    if "parity" in args: parity()
    if "real" in args: build_real()
    if "syn" in args: build_syn([int(a) for a in args if a.isdigit()])
