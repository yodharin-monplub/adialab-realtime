"""Pure-numpy serving version of the Model144 whitened split-test features (no numba in the cloud).

Mirrors models/model144/split_tests.py::row_features step by step:
  AR(p) whitening fitted on the history only (BIC over p in 0,1,2,4,8,12), z-scoring by the history
  residuals, Gaussian mean+variance LLR scan over every candidate split k, two-sample battery between
  the clean reference and the estimated post segment, the same scan on first differences.
One object per series; step(value) returns the 27 features (float32) of the online prefix ending at
`value`.  Deterministic (float64 numpy, fixed operation order).  Parity with the numba training code is
checked by parity_numpy.py.
"""
from __future__ import annotations

import math

import numpy as np

HIST = 1024
MINPOST = 8
MAXPOST = 512
AR_ORDERS = (0, 1, 2, 4, 8, 12)
NAMES = ["sp_llr_max", "sp_llr_k0", "sp_llr_sharp", "sp_khat_rel", "sp_log_post", "sp_age",
         "sp_welch_t", "sp_abs_welch", "sp_logvr", "sp_abs_logvr", "sp_ks", "sp_kuiper", "sp_wasser",
         "sp_mwu_absz", "sp_mad_ratio", "sp_ac1_diff", "sp_kurt_diff", "sp_rough_ratio",
         "spd_llr_max", "spd_llr_k0", "spd_llr_sharp", "spd_khat_rel", "spd_log_post", "spd_age",
         "spd_logvr", "spd_abs_welch", "sp_khat_agree"]
F = len(NAMES)


def fit_ar(history):
    """Identical to models/model57/whitening.py::_fit_ar (history-only AR order selection by BIC)."""
    h = np.asarray(history[-min(len(history), 2048):], dtype=np.float64)
    if len(h) < 64:
        raise ValueError("Insufficient clean history for AR order selection")
    target = h[max(AR_ORDERS):]
    length = len(target)
    best = None
    for order in AR_ORDERS:
        if order:
            lagged = np.column_stack([h[max(AR_ORDERS) - lag:len(h) - lag] for lag in range(1, order + 1)])
            design = np.column_stack((np.ones(length), lagged))
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        else:
            coefficients = np.asarray([float(np.mean(target))])
        residual = target - coefficients[0]
        if order:
            residual -= lagged @ coefficients[1:]
        mse = max(float(np.mean(residual * residual)), 1e-8)
        bic = length * np.log(mse) + (order + 1) * np.log(length)
        if best is None or bic < best[0]:
            best = (bic, order, coefficients.copy())
    _, order, coefficients = best
    coefficients[1:] = np.clip(coefficients[1:], -3, 3)
    return int(order), float(coefficients[0]), coefficients[1:]


def history_residuals(h, order, intercept, phi):
    if order == 0:
        return h - intercept
    windows = np.lib.stride_tricks.sliding_window_view(h, order + 1)
    return windows[:, order] - intercept - windows[:, :order][:, ::-1] @ phi


def _ac1(x):
    m = x.mean(); d = float(((x - m) ** 2).sum())
    return float(((x[1:] - m) * (x[:-1] - m)).sum()) / d if d > 0 else 0.0


def _kurt(x):
    d = x - x.mean(); s2 = float((d * d).sum())
    return len(x) * float((d ** 4).sum()) / (s2 * s2) - 3.0 if s2 > 0 else 0.0


def _rough(x):
    return float(np.abs(np.diff(x)).sum()) / max(len(x) - 1, 1)


def _scan(c1, c2, p, H, sh1, sh2):
    N = H + p
    tot1 = sh1 + c1[p]; tot2 = sh2 + c2[p]
    vall = max(tot2 / N - (tot1 / N) ** 2, 1e-12)
    k = np.arange(0, p - MINPOST + 1)
    n1 = H + k; n2 = p - k
    s1 = sh1 + c1[k]; q1 = sh2 + c2[k]
    s2 = c1[p] - c1[k]; q2 = c2[p] - c2[k]
    v1 = np.maximum(q1 / n1 - (s1 / n1) ** 2, 1e-12); v2 = np.maximum(q2 / n2 - (s2 / n2) ** 2, 1e-12)
    llr = N * math.log(vall) - n1 * np.log(v1) - n2 * np.log(v2)
    bk = int(np.argmax(llr))                       # first maximum, like the strict '>' of the training code
    return float(llr[bk]), float(llr[0]), float(llr.sum() / len(llr)), bk


class SplitTests:
    def __init__(self, history):
        h = np.asarray(history, np.float64)
        self.order, self.intercept, self.phi = fit_ar(h)
        self.recent = list(h[-self.order:]) if self.order else []
        r = history_residuals(h, self.order, self.intercept, self.phi)
        self.hm = float(r.mean()); self.hsd = float(r.std()) + 1e-12
        zfull = (r - self.hm) / self.hsd
        zh = zfull[-HIST:]; dh = np.diff(zfull)[-HIST:]
        self.hs = np.sort(zh); self.H = len(zh); self.sh1 = float(zh.sum()); self.sh2 = float((zh * zh).sum())
        self.hbf = float(np.mean(np.abs(zh - np.median(zh)))) + 1e-12
        self.hac = _ac1(zh); self.hku = _kurt(zh); self.hro = _rough(zh) + 1e-12
        self.Hd = len(dh); self.sd1 = float(dh.sum()); self.sd2 = float((dh * dh).sum())
        self.prev = float(zfull[-1])
        cap = 2048
        self.z = np.zeros(cap); self.c1 = np.zeros(cap + 1); self.c2 = np.zeros(cap + 1)
        self.d1 = np.zeros(cap + 1); self.d2 = np.zeros(cap + 1)
        self.labels = np.concatenate((np.zeros(self.H, np.int8), np.ones(MAXPOST, np.int8)))
        self.state = [-10.0, 0.0, -10.0, 0.0]; self.p = 0

    def _grow(self):
        for name in ("z", "c1", "c2", "d1", "d2"):
            old = getattr(self, name); new = np.zeros(2 * len(old)); new[:len(old)] = old; setattr(self, name, new)

    def step(self, value):
        x = float(value) - self.intercept
        if self.order:
            x -= float(np.asarray(self.recent[::-1]) @ self.phi)
            self.recent = self.recent[1:] + [float(value)]
        if self.p + 2 >= len(self.z):
            self._grow()
        t = self.p; zt = (x - self.hm) / self.hsd; dzt = zt - self.prev; self.prev = zt
        self.z[t] = zt
        self.c1[t + 1] = self.c1[t] + zt; self.c2[t + 1] = self.c2[t] + zt * zt
        self.d1[t + 1] = self.d1[t] + dzt; self.d2[t + 1] = self.d2[t] + dzt * dzt
        self.p = p = t + 1
        out = np.zeros(F)
        if p < MINPOST:
            return out.astype(np.float32)
        c1, c2, H = self.c1, self.c2, self.H
        best, at0, mean_llr, k = _scan(c1, c2, p, H, self.sh1, self.sh2)
        self.state[1] = self.state[1] + 1 if abs(k - self.state[0]) <= 2 else 0.0
        self.state[0] = k
        out[0] = best; out[1] = at0; out[2] = best - mean_llr; out[3] = k / p
        out[4] = math.log1p(p - k); out[5] = math.log1p(self.state[1])
        start = max(k, p - MAXPOST); m = p - start
        n1 = H + k; s1 = self.sh1 + c1[k]; q1 = self.sh2 + c2[k]
        m1 = s1 / n1; v1 = max(q1 / n1 - m1 * m1, 1e-12)
        s2 = c1[p] - c1[start]; q2 = c2[p] - c2[start]
        m2 = s2 / m; v2 = max(q2 / m - m2 * m2, 1e-12)
        tt = (m2 - m1) / math.sqrt(v1 / n1 + v2 / m)
        out[6] = tt; out[7] = abs(tt); out[8] = math.log(v2 / v1); out[9] = abs(out[8])
        w = self.z[start:p]; ws = np.sort(w); n = H
        lab = self.labels[:n + m]
        order = np.lexsort((lab, np.concatenate((self.hs, ws))))          # history first on ties
        lo = lab[order].astype(np.float64)
        fw = np.cumsum(lo) / m; fh = np.cumsum(1.0 - lo) / n
        dplus = max(0.0, float((fw - fh).max())); dminus = max(0.0, float((fh - fw).max()))
        scale = math.sqrt(n * m / (n + m))
        out[10] = max(dplus, dminus) * scale; out[11] = (dplus + dminus) * scale
        idx = np.minimum(n - 1, ((np.arange(m) + 0.5) / m * n).astype(np.int64))
        out[12] = float(np.abs(ws - self.hs[idx]).sum()) / m
        U = float(np.searchsorted(self.hs, ws, side="right").sum())
        out[13] = abs((U - n * m / 2.0) / math.sqrt(n * m * (n + m + 1) / 12.0))
        med = ws[m // 2] if m % 2 == 1 else 0.5 * (ws[m // 2 - 1] + ws[m // 2])
        out[14] = math.log((float(np.abs(w - med).sum()) / m + 1e-12) / self.hbf)
        out[15] = _ac1(w) - self.hac; out[16] = _kurt(w) - self.hku; out[17] = math.log((_rough(w) + 1e-12) / self.hro)
        d1, d2, Hd = self.d1, self.d2, self.Hd
        bestd, at0d, meand, kd = _scan(d1, d2, p, Hd, self.sd1, self.sd2)
        self.state[3] = self.state[3] + 1 if abs(kd - self.state[2]) <= 2 else 0.0
        self.state[2] = kd
        out[18] = bestd; out[19] = at0d; out[20] = bestd - meand; out[21] = kd / p
        out[22] = math.log1p(p - kd); out[23] = math.log1p(self.state[3])
        startd = max(kd, p - MAXPOST); md = p - startd
        n1d = Hd + kd; a1 = self.sd1 + d1[kd]; b1 = self.sd2 + d2[kd]
        mu1 = a1 / n1d; var1 = max(b1 / n1d - mu1 * mu1, 1e-12)
        a2 = d1[p] - d1[startd]; b2 = d2[p] - d2[startd]
        mu2 = a2 / md; var2 = max(b2 / md - mu2 * mu2, 1e-12)
        out[24] = math.log(var2 / var1); out[25] = abs((mu2 - mu1) / math.sqrt(var1 / n1d + var2 / md))
        out[26] = abs(k - kd) / p
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
