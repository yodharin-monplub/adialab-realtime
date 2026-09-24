"""EWMA detector bank with exact finite-horizon null variances.

detectors15 uses an accurate stationary autocovariance correction with a
smooth finite-start ramp.  This variant computes the finite-start variance
of every EWMA exactly from the history-estimated autocovariance sequence via
the recursion

    y_n = alpha * g_n + (1-alpha) * y_{n-1}.

This matters most for short streams and negatively autocorrelated aspects,
where the stationary correction is not yet applicable.  Variance ratios are
clipped for robustness, exactly as in the accepted approximate bank.  All
calibration is fixed from break-free history before online data arrive.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3
import detectors15 as d15


ALPHAS = d15.ALPHAS
MAX_ACOV_LAG = d15.MAX_ACOV_LAG
MIN_RATIO = d15.MIN_RATIO
MAX_RATIO = d15.MAX_RATIO
MAX_ONLINE = 1024
N_ASPECTS = len(d3.ASPECTS)
N_EXTRA = N_ASPECTS * len(ALPHAS) * 2
N_FEATURES = d3.N_FEATURES + N_EXTRA


def feature_names():
    return d3.feature_names() + [name.replace("ema_", "emaexact_", 1)
                                 for name in d15.feature_names()[d3.N_FEATURES:]]


def _autocovariance(g):
    x = np.asarray(g, dtype=np.float64)
    x = x - float(np.mean(x))
    gamma0 = float(np.mean(x * x))
    if x.size < 16 or not np.isfinite(gamma0) or gamma0 < 1e-18:
        out = np.zeros(MAX_ACOV_LAG + 1, dtype=np.float64)
        out[0] = gamma0 if gamma0 > 0 else 1.0
        return out
    max_lag = min(MAX_ACOV_LAG, x.size - 1)
    nfft = 1 << (2 * x.size - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    raw = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:max_lag + 1]
    gamma = raw / np.arange(x.size, x.size - max_lag - 1, -1)
    out = np.zeros(MAX_ACOV_LAG + 1, dtype=np.float64)
    out[:max_lag + 1] = gamma
    return out


class ExactEmaScales:
    def __init__(self, x_hist, hs=None, hist_inc=None):
        if hs is None or hist_inc is None:
            hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        gamma = np.empty((N_ASPECTS, MAX_ACOV_LAG + 1), dtype=np.float64)
        for j, aspect in enumerate(d3.ASPECTS):
            gamma[j] = _autocovariance(hist_inc[aspect] / hs.s[aspect])
        self.gamma = gamma
        self.curve = self._build_curve(MAX_ONLINE)

    def _build_curve(self, length):
        alpha = ALPHAS[None, :]
        r = 1.0 - alpha
        gamma0 = self.gamma[:, 0, None]
        # Keep the estimated covariance recurrence exact.  Robust clipping is
        # applied only to the emitted scale, never fed back into the next step.
        raw_var = alpha * alpha * gamma0
        weighted_cov = np.zeros_like(raw_var)
        power = np.ones_like(alpha)
        curve = np.empty((length, N_ASPECTS, len(ALPHAS)), dtype=np.float64)
        curve[0] = np.sqrt(np.maximum(raw_var, 1e-18))
        for n in range(2, length + 1):
            lag = n - 1
            if lag <= MAX_ACOV_LAG:
                weighted_cov += alpha * power * self.gamma[:, lag, None]
                power *= r
            independent = (gamma0 * alpha / (2.0 - alpha) *
                           (1.0 - r ** (2 * n)))
            raw_var = alpha * alpha * gamma0 + r * r * raw_var \
                + 2.0 * alpha * r * weighted_cov
            ratio = np.clip(raw_var / np.maximum(independent, 1e-18),
                            MIN_RATIO, MAX_RATIO)
            scale_var = independent * ratio
            curve[n - 1] = np.sqrt(np.maximum(scale_var, 1e-18))
        return curve

    def scale(self, n):
        n = int(n)
        if not 1 <= n <= self.curve.shape[0]:
            raise ValueError(f"online step {n} exceeds calibrated horizon")
        return self.curve[n - 1]

    def matrix(self, length):
        length = int(length)
        if not 0 <= length <= self.curve.shape[0]:
            raise ValueError(f"online length {length} exceeds calibrated horizon")
        return self.curve[:length]


def _extra_offline(x_hist, x_online):
    hs, hist_inc, inc = d15._hist_and_online_increments(x_hist, x_online)
    scales = ExactEmaScales(x_hist, hs=hs, hist_inc=hist_inc)
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


class StreamDetector17:
    def __init__(self, x_hist):
        self.inner = d3.StreamDetector3(x_hist)
        self.scales = ExactEmaScales(x_hist)
        self.state = np.zeros((N_ASPECTS, len(ALPHAS)), dtype=np.float64)

    def step(self, x):
        base = self.inner.step(x)
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


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        T = int(rng.integers(5, 700))
        ar = rng.uniform(-0.85, 0.85)
        eps = rng.standard_t(df=rng.uniform(3.0, 12.0), size=n_hist + T)
        x = np.empty(n_hist + T)
        x[0] = eps[0]
        for i in range(1, x.size):
            x[i] = ar * x[i - 1] + eps[i]
        hist, online = x[:n_hist].copy(), x[n_hist:].copy()
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            online[tau:] *= rng.uniform(1.2, 1.8)
        offline = compute_features_offline(hist, online)
        stream = StreamDetector17(hist)
        streamed = np.vstack([stream.step(value) for value in online])
        assert offline.shape == (T, N_FEATURES)
        error = float(np.max(np.abs(offline - streamed)))
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES
    detector = StreamDetector17(rng.normal(size=3000))
    online = rng.normal(size=1000)
    t0 = time.perf_counter()
    for value in online:
        detector.step(value)
    elapsed = (time.perf_counter() - t0) / len(online)
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
