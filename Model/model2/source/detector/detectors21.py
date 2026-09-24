"""Null-calibrated Dirichlet-multinomial distribution-change evidence.

Unlike Pearson/G-test distances, the integrated Bayes factor below pays an
automatic complexity penalty for estimating an alternative categorical
distribution.  Multiple prior concentrations and suffix scales are used.
Every statistic is standardized by rolling values from that series' quiet
history.  Offline and incremental implementations are exact twins.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import gammaln


BIN_COUNTS = (8, 16)
WINDOWS = (4, 8, 16, 32, 64, 128, 256, 512)
CONCENTRATIONS = (1.0, 8.0, 32.0)
N_EXTRA = len(BIN_COUNTS) * (len(WINDOWS) * len(CONCENTRATIONS) +
                             len(CONCENTRATIONS))


def feature_names():
    names = []
    for bins in BIN_COUNTS:
        for window in WINDOWS:
            for concentration in CONCENTRATIONS:
                names.append(f"dm_b{bins}_a{concentration:g}_w{window}")
        for concentration in CONCENTRATIONS:
            names.append(f"dm_b{bins}_a{concentration:g}_scanmax")
    return names


def _bayes_factors(counts, pmf):
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    p = np.maximum(np.asarray(pmf, dtype=np.float64), 1e-12)
    p /= p.sum()
    n = counts.sum(axis=1)
    null = counts @ np.log(p)
    out = np.empty((counts.shape[0], len(CONCENTRATIONS)), dtype=np.float64)
    for j, concentration in enumerate(CONCENTRATIONS):
        alpha = concentration * p
        alternative = (gammaln(concentration) - gammaln(concentration + n) +
                       np.sum(gammaln(counts + alpha[None, :]) -
                              gammaln(alpha[None, :]), axis=1))
        out[:, j] = alternative - null
    return out


class Hist21:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.refs = {}
        for bins in BIN_COUNTS:
            edges = np.unique(np.quantile(
                h, np.arange(1, bins, dtype=np.float64) / bins))
            hb = np.searchsorted(edges, h, side="right")
            n_bins = edges.size + 1
            hist_count = np.bincount(hb, minlength=n_bins).astype(np.float64)
            pmf = hist_count / hist_count.sum()
            onehot = np.zeros((h.size, n_bins), dtype=np.float64)
            onehot[np.arange(h.size), hb] = 1.0
            cumulative = np.vstack((
                np.zeros((1, n_bins), dtype=np.float64),
                np.cumsum(onehot, axis=0)))
            calibration = {}
            for window in WINDOWS:
                counts = cumulative[window:] - cumulative[:-window]
                raw = _bayes_factors(counts, pmf)
                mean = raw.mean(axis=0)
                sd = raw.std(axis=0)
                sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
                calibration[window] = (mean, sd)
            self.refs[bins] = (edges, pmf, calibration)


def extra_features(x_hist, x_online):
    reference = Hist21(x_hist)
    x = np.asarray(x_online, dtype=np.float64)
    T = x.size
    columns = []
    for bins in BIN_COUNTS:
        edges, pmf, calibration = reference.refs[bins]
        online_bins = np.searchsorted(edges, x, side="right")
        onehot = np.zeros((T, pmf.size), dtype=np.float64)
        onehot[np.arange(T), online_bins] = 1.0
        cumulative = np.vstack((
            np.zeros((1, pmf.size), dtype=np.float64),
            np.cumsum(onehot, axis=0)))
        by_concentration = [[] for _ in CONCENTRATIONS]
        for window in WINDOWS:
            z = np.zeros((T, len(CONCENTRATIONS)), dtype=np.float64)
            if T >= window:
                counts = cumulative[window:] - cumulative[:-window]
                raw = _bayes_factors(counts, pmf)
                mean, sd = calibration[window]
                z[window - 1:] = np.clip((raw - mean) / sd, -10.0, 50.0)
            for j in range(len(CONCENTRATIONS)):
                columns.append(z[:, j])
                by_concentration[j].append(z[:, j])
        for values in by_concentration:
            columns.append(np.maximum(np.column_stack(values), 0.0).max(axis=1))
    return np.column_stack(columns)


class StreamDetector21:
    def __init__(self, x_hist):
        self.hist = Hist21(x_hist)
        self.t = 0
        self.max_window = max(WINDOWS)
        self.rings = {}
        self.counts = {}
        for bins in BIN_COUNTS:
            _, pmf, _ = self.hist.refs[bins]
            self.rings[bins] = np.zeros(self.max_window, dtype=np.int64)
            self.counts[bins] = {
                window: np.zeros(pmf.size, dtype=np.float64)
                for window in WINDOWS}

    def step(self, x):
        out = np.empty(N_EXTRA, dtype=np.float64)
        column = 0
        t = self.t
        n = t + 1
        position = t % self.max_window
        for bins in BIN_COUNTS:
            edges, pmf, calibration = self.hist.refs[bins]
            current = int(np.searchsorted(edges, float(x), side="right"))
            ring = self.rings[bins]
            scan = [[] for _ in CONCENTRATIONS]
            for window in WINDOWS:
                count = self.counts[bins][window]
                count[current] += 1.0
                if t >= window:
                    count[ring[(t-window) % self.max_window]] -= 1.0
                if n >= window:
                    raw = _bayes_factors(count, pmf)[0]
                    mean, sd = calibration[window]
                    z = np.clip((raw - mean) / sd, -10.0, 50.0)
                else:
                    z = np.zeros(len(CONCENTRATIONS), dtype=np.float64)
                out[column:column + len(CONCENTRATIONS)] = z
                column += len(CONCENTRATIONS)
                for j, value in enumerate(z):
                    scan[j].append(max(float(value), 0.0))
            for values in scan:
                out[column] = max(values)
                column += 1
            ring[position] = current
        self.t = n
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(21)
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3000))
        T = int(rng.integers(10, 700))
        ar = rng.uniform(-0.8, 0.8)
        eps = rng.standard_t(rng.uniform(3.0, 15.0), n_hist + T)
        values = np.empty(n_hist + T)
        values[0] = eps[0]
        for i in range(1, values.size):
            values[i] = ar * values[i - 1] + eps[i]
        hist, online = values[:n_hist], values[n_hist:].copy()
        if trial % 2 == 0 and T > 20:
            tau = int(rng.integers(5, T - 5))
            post = online[tau:]
            online[tau:] = np.sign(post) * np.sqrt(np.abs(post))
        offline = extra_features(hist, online)
        stream = StreamDetector21(hist)
        streamed = np.vstack([stream.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (T, N_EXTRA)
        assert error < 1e-9, (trial, error)
    detector = StreamDetector21(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_EXTRA
    print(f"OK: detectors21 stream == offline; {N_EXTRA} features; "
          f"{elapsed*1e6:.0f} us/step")
