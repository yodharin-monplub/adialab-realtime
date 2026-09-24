"""Streaming (per-step) version of fast_tests.series_features for serving: numpy only, no numba.

FastTests(history).step(value) returns the 13 features (float32) of the online prefix ending at
`value`, identical (up to float summation order) to fast_tests.series_features(history, online)[t].
Cost per step: O(p) numpy for the prefix scans, O((HIST + W) log) for the window statistics.
"""
from __future__ import annotations

import math

import numpy as np

W = 128
HIST = 1024
WS = 64  # spectral window
NAMES = ["ft_cusum_sq", "ft_var_lr_scan", "ft_chow_f_scan", "ft_buishand", "ft_page_hinkley",
         "ft_levene128", "ft_brown_forsythe128", "ft_abs_log_var_ratio128", "ft_wasserstein128", "ft_kuiper128",
         "ft_ljung_box128", "ft_kurtosis128", "ft_spec_entropy64"]


def _acf_sum(x, lags=10):
    n = len(x); m = x.mean(); d = float(((x - m) ** 2).sum())
    if d <= 0 or n < 4:
        return 0.0
    lags = min(lags, n - 2); q = 0.0; xc = x - m
    for k in range(1, lags + 1):
        r = float((xc[k:] * xc[:-k]).sum()) / d
        q += r * r / (n - k)
    return n * (n + 2) * q / n


def _kurt(x):
    n = len(x); d = x - x.mean(); s2 = float((d * d).sum())
    if s2 <= 0:
        return 0.0
    return n * float((d ** 4).sum()) / (s2 * s2) - 3.0


def _entropy(win):
    p = np.abs(np.fft.rfft(win - win.mean()))[1:] ** 2
    p = p / (p.sum() + 1e-12)
    return float(-(np.where(p > 0, p * np.log(p + 1e-30), 0.0)).sum())


class FastTests:
    def __init__(self, history):
        h = np.asarray(history, dtype=np.float64)
        hist = h[-HIST:] if len(h) > HIST else h
        self.hs = np.sort(hist); self.n_h = len(self.hs)
        self.hm = float(hist.mean()); self.hv = float(hist.var()); self.hsd = math.sqrt(self.hv) + 1e-12
        self.hmad = float(np.mean(np.abs(hist - self.hm))) + 1e-12
        hmed = float(np.median(hist)); self.hbf = float(np.mean(np.abs(hist - hmed))) + 1e-12
        self.hlb = _acf_sum(hist[-W:]) if len(hist) >= W else 0.0
        self.hk = _kurt(hist[-W:]) if len(hist) >= W else 0.0
        self.href = _entropy(h[-WS:]) if len(h) >= WS else 0.0
        self.labels = np.concatenate((np.zeros(self.n_h, np.int8), np.ones(W, np.int8)))  # merge labels (hs first on ties)
        cap = 4096
        self.o = np.empty(cap); self.c1 = np.zeros(cap + 1); self.c2 = np.zeros(cap + 1)
        self.p = 0
        self.cs_max = 0.0; self.cs_min = 0.0
        self.ph = 0.0; self.ph_max = 0.0; self.ph_min = 0.0

    def _grow(self):
        cap = 2 * len(self.o)
        o = np.empty(cap); o[:self.p] = self.o[:self.p]; self.o = o
        for name in ("c1", "c2"):
            old = getattr(self, name); new = np.zeros(cap + 1); new[:self.p + 1] = old[:self.p + 1]; setattr(self, name, new)

    def step(self, value):
        if self.p == len(self.o):
            self._grow()
        x = float(value); t = self.p; p = t + 1
        self.o[t] = x; self.c1[p] = self.c1[t] + x; self.c2[p] = self.c2[t] + x * x
        self.p = p
        out = np.zeros(13, dtype=np.float64)
        c1, c2, hv = self.c1, self.c2, self.hv
        # prefix scans
        v = c2[p] - hv * p
        if v > self.cs_max: self.cs_max = v
        if v < self.cs_min: self.cs_min = v
        out[0] = max(self.cs_max, -self.cs_min) / (hv * math.sqrt(p) + 1e-12)
        k = np.arange(1, p + 1, dtype=np.float64)
        b = c1[1:p + 1] - k * (c1[p] / p)
        out[3] = (b.max() - b.min()) / (self.hsd * math.sqrt(p))
        self.ph += x - self.hm - 0.05 * self.hsd
        if self.ph > self.ph_max: self.ph_max = self.ph
        if self.ph < self.ph_min: self.ph_min = self.ph
        out[4] = (self.ph_max - self.ph_min) / self.hsd
        if p >= 8:
            kk = np.arange(4, p - 3, dtype=np.float64); ck1 = c1[4:p - 3]; ck2 = c2[4:p - 3]
            v1 = ck2 / kk; v2 = (c2[p] - ck2) / (p - kk); vv = c2[p] / p
            ll = p * math.log(vv + 1e-12) - kk * np.log(v1 + 1e-12) - (p - kk) * np.log(v2 + 1e-12)
            out[1] = max(0.0, float(ll.max()))
            m1 = ck1 / kk; m2 = (c1[p] - ck1) / (p - kk)
            f = kk * (p - kk) / p * (m1 - m2) ** 2
            out[2] = max(0.0, float(f.max())) / (hv + 1e-12)
        # window 128 vs history
        lo = max(0, p - W); w = self.o[lo:p]
        if len(w) >= 8:
            wm = w.mean()
            out[5] = math.log((float(np.mean(np.abs(w - wm))) + 1e-12) / self.hmad)
            out[6] = math.log((float(np.mean(np.abs(w - np.median(w)))) + 1e-12) / self.hbf)
            out[7] = abs(math.log((float(w.var()) + 1e-12) / (hv + 1e-12)))
            ws = np.sort(w); m = len(ws); n = self.n_h
            q = (np.arange(m) + 0.5) / m
            hq = self.hs[np.minimum(n - 1, (q * n).astype(np.int64))]
            out[8] = float(np.mean(np.abs(ws - hq)))
            values = np.concatenate((self.hs, ws)); labels = self.labels[:n + m]
            order = np.lexsort((labels, values))
            lab = labels[order].astype(np.float64)
            fw = np.cumsum(lab) / m; fh = np.cumsum(1.0 - lab) / n
            out[9] = float((fw - fh).max()) + float((fh - fw).max())
            out[10] = _acf_sum(w) - self.hlb
            out[11] = _kurt(w) - self.hk
        if p >= WS:
            out[12] = _entropy(self.o[p - WS:p]) - self.href
        return out.astype(np.float32)
