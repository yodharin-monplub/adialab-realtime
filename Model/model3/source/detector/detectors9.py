"""detectors3 plus per-series long-run-variance calibrated copies.

The original detector bank divides cumulative statistics by the historical
one-step increment standard deviation.  For serially dependent increments,
Var(sum(g, w)) is not w*Var(g), so equally sized statistics need not have
comparable null distributions across series.  This appends HAC-calibrated
copies while retaining every original feature.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3


MAX_HAC_LAG = 32
MIN_VAR_RATIO = 0.25
MAX_VAR_RATIO = 4.0
BLOCK = 3 + len(d3.WINDOWS)
N_HAC_DYNAMIC = len(d3.ASPECTS) * BLOCK
N_HAC_CONTEXT = len(d3.ASPECTS)
N_FEATURES = d3.N_FEATURES + N_HAC_DYNAMIC + N_HAC_CONTEXT


def feature_names():
    base = d3.feature_names()
    dynamic = base[:len(d3.ASPECTS) * BLOCK]
    return (base + [f"hac_{name}" for name in dynamic]
            + [f"haclog_{a}" for a in d3.ASPECTS])


def _hist_increments(x_hist):
    h = np.asarray(x_hist, dtype=np.float64)
    hs = d3.Hist3(h)
    z = (h - hs.mu0) / hs.sd0
    u = hs._cdf(h)
    e = d3._ar_residuals(z, hs.phi, np.zeros(d3.AR_P))
    e = e[d3.AR_P:] if e.size > d3.AR_P else e
    if e.size < 10:
        e = z
    return d3._aspect_increments(hs, z, u, eh_pair=(e, None))


def _variance_ratio(g, window):
    x = np.asarray(g, dtype=np.float64)
    x = x - float(np.mean(x))
    gamma0 = float(np.mean(x * x))
    if x.size < 4 or not np.isfinite(gamma0) or gamma0 < 1e-18:
        return 1.0
    max_lag = min(MAX_HAC_LAG, int(window) - 1, x.size - 1)
    ratio = 1.0
    for lag in range(1, max_lag + 1):
        gamma = float(np.mean(x[lag:] * x[:-lag]))
        ratio += 2.0 * (1.0 - lag / float(window)) * gamma / gamma0
    if not np.isfinite(ratio):
        ratio = 1.0
    return float(np.clip(ratio, MIN_VAR_RATIO, MAX_VAR_RATIO))


class HACScales:
    def __init__(self, x_hist):
        inc = _hist_increments(x_hist)
        self.mult = np.empty((len(d3.ASPECTS), BLOCK), dtype=np.float64)
        self.log_context = np.empty(len(d3.ASPECTS), dtype=np.float64)
        for j, aspect in enumerate(d3.ASPECTS):
            ratios = [_variance_ratio(inc[aspect], w) for w in d3.WINDOWS]
            prefix_ratio = ratios[-1]
            self.mult[j, :3] = 1.0 / np.sqrt(prefix_ratio)
            self.mult[j, 3:] = 1.0 / np.sqrt(ratios)
            self.log_context[j] = np.log(prefix_ratio)

    def append(self, base):
        base = np.asarray(base, dtype=np.float64)
        T = base.shape[0]
        corrected = np.empty((T, N_HAC_DYNAMIC), dtype=np.float64)
        for j in range(len(d3.ASPECTS)):
            lo = j * BLOCK
            hi = lo + BLOCK
            corrected[:, lo:hi] = base[:, lo:hi] * self.mult[j]
        context = np.broadcast_to(self.log_context, (T, N_HAC_CONTEXT))
        return np.column_stack([base, corrected, context])


def augment_base_features(x_hist, base_features):
    return HACScales(x_hist).append(base_features)


def compute_features_offline(x_hist, x_online):
    base = d3.compute_features_offline(x_hist, x_online)
    return augment_base_features(x_hist, base)


class StreamDetector9:
    def __init__(self, x_hist):
        self.inner = d3.StreamDetector3(x_hist)
        scales = HACScales(x_hist)
        self.mult_flat = scales.mult.ravel()
        self.log_context = scales.log_context

    def step(self, x):
        base = self.inner.step(x)
        out = np.empty(N_FEATURES, dtype=np.float64)
        out[:d3.N_FEATURES] = base
        lo = d3.N_FEATURES
        hi = lo + N_HAC_DYNAMIC
        out[lo:hi] = base[:N_HAC_DYNAMIC] * self.mult_flat
        out[hi:] = self.log_context
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    for trial in range(10):
        n_hist = int(rng.integers(100, 700))
        T = int(rng.integers(5, 700))
        ar = rng.uniform(-0.75, 0.75)
        eps = rng.standard_t(df=rng.uniform(3.0, 12.0), size=n_hist + T)
        x = np.empty(n_hist + T)
        x[0] = eps[0]
        for i in range(1, x.size):
            x[i] = ar * x[i - 1] + eps[i]
        hist = x[:n_hist].copy()
        online = x[n_hist:].copy()
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            if trial % 4 == 0:
                online[tau:] += rng.uniform(0.3, 1.2)
            else:
                online[tau:] *= rng.uniform(1.3, 2.0)
        offline = compute_features_offline(hist, online)
        stream = StreamDetector9(hist)
        streamed = np.vstack([stream.step(v) for v in online])
        assert offline.shape == (T, N_FEATURES)
        error = float(np.max(np.abs(offline - streamed)))
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES
    det = StreamDetector9(rng.normal(size=3000))
    online = rng.normal(size=1000)
    t0 = time.perf_counter()
    for value in online:
        det.step(value)
    elapsed = (time.perf_counter() - t0) / len(online)
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
