"""History-calibrated exponentially weighted copies of detectors3 aspects.

All null normalization uses only the guaranteed break-free history.  The
vectorized offline path and incremental streaming path are independent and
verified by the package preflight.
"""

from __future__ import annotations

import numpy as np

import detectors3 as d3


ALPHAS = np.asarray((0.5, 0.25, 0.125, 0.0625,
                     0.03125, 0.015625, 0.0078125), dtype=np.float64)
MAX_ACOV_LAG = 512
MIN_RATIO = 0.10
MAX_RATIO = 10.0
N_ASPECTS = len(d3.ASPECTS)
N_EXTRA = N_ASPECTS * len(ALPHAS) * 2
N_FEATURES = d3.N_FEATURES + N_EXTRA


def feature_names():
    extra = []
    for aspect in d3.ASPECTS:
        for alpha in ALPHAS:
            tag = f"a{alpha:.7f}".rstrip("0").rstrip(".")
            extra.extend((f"ema_{aspect}_{tag}", f"absema_{aspect}_{tag}"))
    return d3.feature_names() + extra


def _hist_and_online_increments(x_hist, x_online=None):
    h = np.asarray(x_hist, dtype=np.float64)
    hs = d3.Hist3(h)
    zh = (h - hs.mu0) / hs.sd0
    uh = hs._cdf(h)
    eh = d3._ar_residuals(zh, hs.phi, np.zeros(d3.AR_P))
    eh = eh[d3.AR_P:] if eh.size > d3.AR_P else eh
    if eh.size < 10:
        eh = zh
    hist_inc = d3._aspect_increments(hs, zh, uh, eh_pair=(eh, None))
    if x_online is None:
        return hs, hist_inc, None
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hs.mu0) / hs.sd0
    u = hs._cdf(x)
    e = d3._ar_residuals(z, hs.phi, hs.zh_tail)
    online_inc = d3._aspect_increments(hs, z, u, e=e)
    return hs, hist_inc, online_inc


def _steady_ratios(g):
    x = np.asarray(g, dtype=np.float64)
    x = x - float(np.mean(x))
    gamma0 = float(np.mean(x * x))
    if x.size < 16 or not np.isfinite(gamma0) or gamma0 < 1e-18:
        return gamma0 if gamma0 > 0 else 1.0, np.ones(len(ALPHAS))
    max_lag = min(MAX_ACOV_LAG, x.size - 1)
    nfft = 1 << (2 * x.size - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    raw = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:max_lag + 1]
    gamma = raw / np.arange(x.size, x.size - max_lag - 1, -1)
    lag = np.arange(1, max_lag + 1, dtype=np.float64)
    ratios = np.empty(len(ALPHAS), dtype=np.float64)
    for k, alpha in enumerate(ALPHAS):
        r = 1.0 - alpha
        ratio = 1.0 + 2.0 * np.sum((r ** lag) * gamma[1:]) / gamma0
        if not np.isfinite(ratio):
            ratio = 1.0
        ratios[k] = np.clip(ratio, MIN_RATIO, MAX_RATIO)
    return gamma0, ratios


class EmaScales:
    def __init__(self, x_hist, hs=None, hist_inc=None):
        if hs is None or hist_inc is None:
            hs, hist_inc, _ = _hist_and_online_increments(x_hist)
        self.var1 = np.empty(N_ASPECTS, dtype=np.float64)
        self.ratio = np.empty((N_ASPECTS, len(ALPHAS)), dtype=np.float64)
        for j, aspect in enumerate(d3.ASPECTS):
            g = hist_inc[aspect] / hs.s[aspect]
            self.var1[j], self.ratio[j] = _steady_ratios(g)

    def scale(self, n):
        n = int(n)
        r = 1.0 - ALPHAS
        independent = (self.var1[:, None] * ALPHAS[None, :] /
                       (2.0 - ALPHAS[None, :]) *
                       (1.0 - r[None, :] ** (2 * n)))
        ramp = 1.0 + (self.ratio - 1.0) * (
            1.0 - r[None, :] ** max(n - 1, 0))
        return np.sqrt(np.maximum(independent * ramp, 1e-18))

    def matrix(self, length):
        n = np.arange(1, int(length) + 1, dtype=np.float64)
        r = 1.0 - ALPHAS
        independent = (self.var1[None, :, None] * ALPHAS[None, None, :] /
                       (2.0 - ALPHAS[None, None, :]) *
                       (1.0 - r[None, None, :] ** (2 * n[:, None, None])))
        ramp = 1.0 + (self.ratio[None, :, :] - 1.0) * (
            1.0 - r[None, None, :] ** np.maximum(n[:, None, None] - 1, 0))
        return np.sqrt(np.maximum(independent * ramp, 1e-18))


def _extra_offline(x_hist, x_online):
    hs, hist_inc, inc = _hist_and_online_increments(x_hist, x_online)
    scales = EmaScales(x_hist, hs=hs, hist_inc=hist_inc)
    T = len(x_online)
    state = np.zeros((N_ASPECTS, len(ALPHAS)), dtype=np.float64)
    states = np.empty((T, N_ASPECTS, len(ALPHAS)), dtype=np.float64)
    r = 1.0 - ALPHAS
    G = np.column_stack([inc[aspect] / hs.s[aspect]
                         for aspect in d3.ASPECTS])
    for t in range(T):
        state = r[None, :] * state + ALPHAS[None, :] * G[t, :, None]
        states[t] = state
    z = states / scales.matrix(T)
    return np.stack([z, np.abs(z)], axis=-1).reshape(T, N_EXTRA)


def extra_features(x_hist, x_online):
    return _extra_offline(x_hist, x_online)


def compute_features_offline(x_hist, x_online):
    base = d3.compute_features_offline(x_hist, x_online)
    return np.column_stack([base, _extra_offline(x_hist, x_online)])


class StreamDetector15:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.scales = EmaScales(x_hist)
        self.state = np.zeros((N_ASPECTS, len(ALPHAS)), dtype=np.float64)

    def _from_current_inner(self, base):
        n = self.inner.t
        pos = (n - 1) % self.inner._wmax
        r = 1.0 - ALPHAS
        for j, aspect in enumerate(d3.ASPECTS):
            g = self.inner._ring[aspect][pos] / self.inner.hs.s[aspect]
            self.state[j] = r * self.state[j] + ALPHAS * g
        z = self.state / self.scales.scale(n)
        pair = np.empty((N_ASPECTS, len(ALPHAS), 2), dtype=np.float64)
        pair[:, :, 0] = z
        pair[:, :, 1] = np.abs(z)
        out = np.empty(N_FEATURES, dtype=np.float64)
        out[:d3.N_FEATURES] = base
        out[d3.N_FEATURES:] = pair.ravel()
        return out

    def step(self, x):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        return self._from_current_inner(self.inner.step(x))

    def step_from_inner(self, base):
        return self._from_current_inner(base)
