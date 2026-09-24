"""detectors3 plus empirically block-calibrated suffix statistics.

The base detector divides a length-w suffix sum by ``sqrt(w)`` times the
one-step historical standard deviation.  That is comparable across series
only when the aspect increments are effectively independent.  detectors9
uses a truncated autocovariance correction; this module estimates the null
variance directly from all overlapping length-w blocks in the break-free
history.

For every detectors3 aspect and geometric suffix window, append a copy of
the suffix statistic multiplied by

    sqrt(w * Var(g_hist) / Var(sum_w(g_hist))).

The block variance is computed from the full historical autocovariance
sequence (via one FFT per aspect), unlike detectors9's 32-lag truncation.
The multiplier is fixed before the online stream starts.  At early online
times, where the effective suffix is shorter than its nominal window, use
the largest calibrated geometric window no longer than the observed prefix
(and no correction before eight observations).  Log variance ratios are
also appended as static context.  Both paths intentionally build on the
independently tested detectors3 offline/stream implementations.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3


MIN_RATIO = 0.10
MAX_RATIO = 10.0
N_ASPECTS = len(d3.ASPECTS)
N_WINDOWS = len(d3.WINDOWS)
BLOCK = 3 + N_WINDOWS
N_CORRECTED = N_ASPECTS * N_WINDOWS
N_CONTEXT = N_CORRECTED
N_FEATURES = d3.N_FEATURES + N_CORRECTED + N_CONTEXT


def feature_names():
    base = d3.feature_names()
    corrected = [f"emp_{a}_w{w}" for a in d3.ASPECTS for w in d3.WINDOWS]
    context = [f"emplog_{a}_w{w}" for a in d3.ASPECTS for w in d3.WINDOWS]
    return base + corrected + context


def _hist_increments(x_hist):
    """Recreate exactly the history increments used to scale detectors3."""
    h = np.asarray(x_hist, dtype=np.float64)
    hs = d3.Hist3(h)
    z = (h - hs.mu0) / hs.sd0
    u = hs._cdf(h)
    e = d3._ar_residuals(z, hs.phi, np.zeros(d3.AR_P))
    e = e[d3.AR_P:] if e.size > d3.AR_P else e
    if e.size < 10:
        e = z
    return d3._aspect_increments(hs, z, u, eh_pair=(e, None))


def _block_ratios(g):
    """Full-autocovariance Var(block sum) / (w * Var(one step))."""
    x = np.asarray(g, dtype=np.float64)
    x = x - float(np.mean(x))
    var1 = float(np.mean(x * x))
    if x.size < 16 or not np.isfinite(var1) or var1 < 1e-18:
        return np.ones(N_WINDOWS, dtype=np.float64)
    max_lag = min(max(d3.WINDOWS) - 1, x.size - 1)
    nfft = 1 << (2 * x.size - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    raw = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:max_lag + 1]
    gamma = raw / np.arange(x.size, x.size - max_lag - 1, -1)
    out = np.ones(N_WINDOWS, dtype=np.float64)
    for k, window in enumerate(d3.WINDOWS):
        w = min(int(window), x.size)
        lag = np.arange(1, w, dtype=np.float64)
        ratio = 1.0 + 2.0 * np.sum((1.0 - lag / w) * gamma[1:w]) / var1
        if not np.isfinite(ratio):
            ratio = 1.0
        out[k] = np.clip(ratio, MIN_RATIO, MAX_RATIO)
    return out


class EmpiricalScales:
    def __init__(self, x_hist):
        inc = _hist_increments(x_hist)
        self.ratio = np.empty((N_ASPECTS, N_WINDOWS), dtype=np.float64)
        for j, aspect in enumerate(d3.ASPECTS):
            self.ratio[j] = _block_ratios(inc[aspect])
        self.mult = 1.0 / np.sqrt(self.ratio)
        self.log_context = np.log(self.ratio).ravel()


def _multiplier_rows(mult, n):
    """Per-time multiplier for each nominal suffix window."""
    n = np.asarray(n, dtype=np.int64)
    windows = np.asarray(d3.WINDOWS)
    effective = np.minimum(n[:, None], windows[None, :])
    # largest calibrated window <= effective; none before w=8
    bucket = np.searchsorted(windows, effective, side="right") - 1
    clipped = np.maximum(bucket, 0)
    # Advanced indexing gives (aspect, time, nominal-window).
    out = mult[:, clipped].transpose(1, 0, 2)
    out[:, :, :][np.broadcast_to(bucket[:, None, :] < 0, out.shape)] = 1.0
    return out.reshape(n.size, N_CORRECTED)


def extra_features(x_hist, base_features):
    base = np.asarray(base_features, dtype=np.float64)
    T = base.shape[0]
    scales = EmpiricalScales(x_hist)
    suffix_cols = []
    block = 3 + N_WINDOWS
    for j in range(N_ASPECTS):
        lo = j * block + 3
        suffix_cols.extend(range(lo, lo + N_WINDOWS))
    suffix = base[:, suffix_cols]
    corrected = suffix * _multiplier_rows(scales.mult, np.arange(1, T + 1))
    context = np.broadcast_to(scales.log_context, (T, N_CONTEXT))
    return np.column_stack([corrected, context])


def augment_base_features(x_hist, base_features):
    base = np.asarray(base_features, dtype=np.float64)
    return np.column_stack([base, extra_features(x_hist, base)])


def compute_features_offline(x_hist, x_online):
    return augment_base_features(
        x_hist, d3.compute_features_offline(x_hist, x_online))


class StreamDetector13:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        scales = EmpiricalScales(x_hist)
        self.mult = scales.mult
        self.log_context = scales.log_context

    def _extra_from_base(self, base):
        n = self.inner.t
        windows = np.asarray(d3.WINDOWS, dtype=np.int64)
        bucket = np.searchsorted(
            windows, np.minimum(n, windows), side="right"
        ) - 1
        clipped = np.maximum(bucket, 0)
        multiplier = self.mult[:, clipped].copy()
        multiplier[:, bucket < 0] = 1.0
        suffix = np.asarray(base, dtype=np.float64)[
            : N_ASPECTS * BLOCK
        ].reshape(N_ASPECTS, BLOCK)[:, 3:]
        out = np.empty(N_CORRECTED + N_CONTEXT, dtype=np.float64)
        out[:N_CORRECTED] = (suffix * multiplier).ravel()
        out[N_CORRECTED:] = self.log_context
        return out

    def step(self, x):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        base = self.inner.step(x)
        return np.concatenate((base, self._extra_from_base(base)))

    def step_from_inner(self, base):
        return self._extra_from_base(base)


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3000))
        T = int(rng.integers(5, 700))
        ar = rng.uniform(-0.8, 0.8)
        eps = rng.standard_t(df=rng.uniform(3.0, 12.0), size=n_hist + T)
        xs = np.empty(n_hist + T)
        xs[0] = eps[0]
        for q in range(1, xs.size):
            xs[q] = ar * xs[q - 1] + eps[q]
        hist, online = xs[:n_hist].copy(), xs[n_hist:].copy()
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            online[tau:] += rng.uniform(0.3, 1.2)

        offline = compute_features_offline(hist, online)
        stream = StreamDetector13(hist)
        streamed = np.vstack([stream.step(v) for v in online])
        assert offline.shape == (T, N_FEATURES)
        error = float(np.max(np.abs(offline - streamed)))
        assert error < 1e-9, (trial, error)

    assert len(feature_names()) == N_FEATURES
    det = StreamDetector13(rng.normal(size=3000))
    online = rng.normal(size=1000)
    t0 = time.perf_counter()
    for value in online:
        det.step(value)
    elapsed = (time.perf_counter() - t0) / len(online)
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
