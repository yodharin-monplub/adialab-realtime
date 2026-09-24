"""detectors3 plus null-calibrated multi-scale histogram change tests.

The largest OOF weakness is early distribution-dominant breaks.  This bank
maps observations to bins defined by the guaranteed break-free historical
ECDF, then computes Pearson chi-square, multinomial G-test, and KS statistics
on suffix windows 8..512.  Crucially, every statistic/window is standardized
by its rolling null mean and standard deviation measured on that same
series' history, preserving cross-series comparability under dependence and
heavy tails.

All detectors3 columns remain first and unchanged.  Offline and streaming
implementations are exact twins and never use future online observations.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3


BIN_COUNTS = (8, 16)
WINDOWS = d3.WINDOWS
STATS = ("pearson", "gtest", "ks")
CLIP_LO, CLIP_HI = -10.0, 50.0

EXTRA_NAMES = []
for bins in BIN_COUNTS:
    for window in WINDOWS:
        EXTRA_NAMES += [f"hist{bins}_{stat}_w{window}" for stat in STATS]
    EXTRA_NAMES += [f"hist{bins}_{stat}_scanmax" for stat in STATS]

N_FEATURES = d3.N_FEATURES + len(EXTRA_NAMES)


def feature_names():
    return d3.feature_names() + EXTRA_NAMES


def _raw_stats(counts, pmf):
    """Pearson, G-test, KS for count rows against a fixed reference PMF."""
    C = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    n = C.sum(axis=1)
    expected = n[:, None] * pmf[None, :]
    safe = np.maximum(expected, 1e-12)
    pearson = np.sum((C - expected) ** 2 / safe, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(C > 0, C * np.log(C / safe), 0.0)
    gtest = 2.0 * terms.sum(axis=1)
    cdf = np.cumsum(C, axis=1) / np.maximum(n[:, None], 1.0)
    ks = np.max(np.abs(cdf - np.cumsum(pmf)[None, :]), axis=1)
    return np.column_stack([pearson, gtest, ks])


class Hist10:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.refs = {}
        for bins in BIN_COUNTS:
            quantiles = np.arange(1, bins, dtype=np.float64) / bins
            edges = np.unique(np.quantile(h, quantiles))
            hb = np.searchsorted(edges, h, side="right")
            n_bins = edges.size + 1
            hist_count = np.bincount(hb, minlength=n_bins).astype(np.float64)
            pmf = hist_count / max(hist_count.sum(), 1.0)

            onehot = np.zeros((h.size, n_bins), dtype=np.float64)
            onehot[np.arange(h.size), hb] = 1.0
            cum = np.vstack([np.zeros((1, n_bins)), np.cumsum(onehot, axis=0)])
            calibration = {}
            for window in WINDOWS:
                if h.size >= window:
                    counts = cum[window:] - cum[:-window]
                    raw = _raw_stats(counts, pmf)
                    mu = raw.mean(axis=0)
                    sd = raw.std(axis=0)
                else:  # histories are >=1000 in this competition
                    mu = np.zeros(len(STATS))
                    sd = np.ones(len(STATS))
                sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
                calibration[window] = (mu, sd)
            self.refs[bins] = (edges, pmf, calibration)


def _standardize(raw, mu, sd):
    return np.clip((raw - mu) / sd, CLIP_LO, CLIP_HI)


def _extras_offline(hist_ref, x_online):
    x = np.asarray(x_online, dtype=np.float64)
    T = x.size
    cols = []
    for bins in BIN_COUNTS:
        edges, pmf, calibration = hist_ref.refs[bins]
        online_bins = np.searchsorted(edges, x, side="right")
        onehot = np.zeros((T, pmf.size), dtype=np.float64)
        onehot[np.arange(T), online_bins] = 1.0
        cum = np.vstack([np.zeros((1, pmf.size)), np.cumsum(onehot, axis=0)])
        by_stat = [[] for _ in STATS]
        for window in WINDOWS:
            z = np.zeros((T, len(STATS)), dtype=np.float64)
            if T >= window:
                counts = cum[window:] - cum[:-window]
                raw = _raw_stats(counts, pmf)
                mu, sd = calibration[window]
                z[window - 1:] = _standardize(raw, mu, sd)
            for k in range(len(STATS)):
                cols.append(z[:, k])
                by_stat[k].append(z[:, k])
        for k in range(len(STATS)):
            cols.append(np.maximum(np.column_stack(by_stat[k]), 0.0).max(axis=1))
    return np.column_stack(cols)


def augment_base_features(x_hist, x_online, base_features):
    extras = _extras_offline(Hist10(x_hist), x_online)
    return np.column_stack([base_features, extras])


def compute_features_offline(x_hist, x_online):
    base = d3.compute_features_offline(x_hist, x_online)
    return augment_base_features(x_hist, x_online, base)


class StreamDetector10:
    def __init__(self, x_hist):
        self.inner = d3.StreamDetector3(x_hist)
        self.hist = Hist10(x_hist)
        self.t = 0
        self.max_window = max(WINDOWS)
        self.rings = {}
        self.counts = {}
        for bins in BIN_COUNTS:
            _, pmf, _ = self.hist.refs[bins]
            self.rings[bins] = np.zeros(self.max_window, dtype=np.int64)
            self.counts[bins] = {
                w: np.zeros(pmf.size, dtype=np.float64) for w in WINDOWS}

    def step(self, x):
        base = self.inner.step(x)
        out = np.empty(N_FEATURES, dtype=np.float64)
        out[:d3.N_FEATURES] = base
        col = d3.N_FEATURES
        pos = self.t % self.max_window
        n = self.t + 1
        for bins in BIN_COUNTS:
            edges, pmf, calibration = self.hist.refs[bins]
            current = int(np.searchsorted(edges, float(x), side="right"))
            ring = self.rings[bins]
            scan = [[] for _ in STATS]
            for window in WINDOWS:
                count = self.counts[bins][window]
                count[current] += 1.0
                if self.t >= window:
                    count[ring[(self.t - window) % self.max_window]] -= 1.0
                if n >= window:
                    raw = _raw_stats(count, pmf)[0]
                    mu, sd = calibration[window]
                    z = _standardize(raw, mu, sd)
                else:
                    z = np.zeros(len(STATS), dtype=np.float64)
                out[col:col + len(STATS)] = z
                col += len(STATS)
                for k in range(len(STATS)):
                    scan[k].append(max(float(z[k]), 0.0))
            for k in range(len(STATS)):
                out[col] = max(scan[k])
                col += 1
            ring[pos] = current
        self.t += 1
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    for trial in range(10):
        n_hist = int(rng.integers(1000, 2500))
        T = int(rng.integers(10, 700))
        ar = rng.uniform(-0.75, 0.75)
        eps = rng.standard_t(df=rng.uniform(3.0, 15.0), size=n_hist + T)
        x = np.empty(n_hist + T)
        x[0] = eps[0]
        for i in range(1, x.size):
            x[i] = ar * x[i - 1] + eps[i]
        hist = x[:n_hist].copy()
        online = x[n_hist:].copy()
        if trial % 2 == 0 and T > 20:
            tau = int(rng.integers(5, T - 5))
            # Non-Gaussian shape change with limited mean/variance signal.
            post = online[tau:]
            online[tau:] = np.sign(post) * np.sqrt(np.abs(post))
        offline = compute_features_offline(hist, online)
        detector = StreamDetector10(hist)
        streamed = np.vstack([detector.step(v) for v in online])
        assert offline.shape == (T, N_FEATURES)
        error = float(np.max(np.abs(offline - streamed)))
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES
    detector = StreamDetector10(rng.normal(size=3000))
    values = rng.normal(size=1000)
    t0 = time.perf_counter()
    for value in values:
        detector.step(value)
    elapsed = (time.perf_counter() - t0) / len(values)
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
