"""detectors3 plus directional Fourier modes of the historical ECDF rank.

Unsigned KS/chi-square distances discard the direction of a distribution
change.  Here u_t = F_hist(x_t) is projected onto sine/cosine modes 1..8.
Each directional channel is centered and scaled on the guaranteed break-free
history, then accumulated with detectors3's prefix/CUSUM/suffix-window bank.
The Fourier basis complements the existing low-order Legendre channels while
remaining O(1) per mode and exactly streamable.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3


MODES = tuple(range(1, 9))
WINDOWS = d3.WINDOWS
CHANNELS = tuple((kind, k) for k in MODES for kind in ("sin", "cos"))
BLOCK = 3 + len(WINDOWS)
N_EXTRA = len(CHANNELS) * BLOCK
N_FEATURES = d3.N_FEATURES + N_EXTRA


def feature_names():
    names = d3.feature_names()
    for kind, k in CHANNELS:
        prefix = f"F{kind[0]}{k}"
        names += [f"{prefix}_prefix", f"{prefix}_up", f"{prefix}_down"]
        names += [f"{prefix}_w{w}" for w in WINDOWS]
    return names


def _channel_values(u):
    u = np.asarray(u, dtype=np.float64)
    cols = []
    for kind, k in CHANNELS:
        phase = 2.0 * np.pi * k * u
        cols.append(np.sin(phase) if kind == "sin" else np.cos(phase))
    return np.column_stack(cols)


class Hist11:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.base = d3.Hist3(h)
        hist_channels = _channel_values(self.base._cdf(h))
        self.mean = hist_channels.mean(axis=0)
        sd = hist_channels.std(axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)


def _extras_offline(hist_ref, x_online):
    u = hist_ref.base._cdf(np.asarray(x_online, dtype=np.float64))
    G = _channel_values(u) - hist_ref.mean[None, :]
    T = G.shape[0]
    n = np.arange(1, T + 1, dtype=np.float64)
    sqrt_n = np.sqrt(n)
    idx = np.arange(1, T + 1)
    cols = []
    for j in range(G.shape[1]):
        g = G[:, j]
        sd = hist_ref.sd[j]
        cumulative = np.cumsum(g)
        padded = np.concatenate([[0.0], cumulative])
        run_min = np.minimum.accumulate(np.minimum(cumulative, 0.0))
        run_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        cols.append(cumulative / (sd * sqrt_n))
        cols.append((cumulative - run_min) / (sd * sqrt_n))
        cols.append((run_max - cumulative) / (sd * sqrt_n))
        for window in WINDOWS:
            start = np.maximum(0, idx - window)
            effective = np.minimum(n, window)
            total = padded[1:] - padded[start]
            cols.append(total / (sd * np.sqrt(effective)))
    return np.column_stack(cols)


def augment_base_features(x_hist, x_online, base_features):
    extras = _extras_offline(Hist11(x_hist), x_online)
    return np.column_stack([base_features, extras])


def compute_features_offline(x_hist, x_online):
    base = d3.compute_features_offline(x_hist, x_online)
    return augment_base_features(x_hist, x_online, base)


class StreamDetector11:
    def __init__(self, x_hist):
        self.inner = d3.StreamDetector3(x_hist)
        h = np.asarray(x_hist, dtype=np.float64)
        hist_channels = _channel_values(self.inner.hs._cdf(h))
        self.mean = hist_channels.mean(axis=0)
        sd = hist_channels.std(axis=0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        self.t = 0
        self.cumulative = np.zeros(len(CHANNELS), dtype=np.float64)
        self.run_min = np.zeros(len(CHANNELS), dtype=np.float64)
        self.run_max = np.zeros(len(CHANNELS), dtype=np.float64)
        self.ring = np.zeros((max(WINDOWS), len(CHANNELS)), dtype=np.float64)
        self.window_sum = {
            w: np.zeros(len(CHANNELS), dtype=np.float64) for w in WINDOWS}

    def step(self, x):
        base = self.inner.step(x)
        u = self.inner.hs._cdf(np.asarray([float(x)]))[0]
        g = _channel_values(np.asarray([u]))[0] - self.mean
        n = self.t + 1
        pos = self.t % max(WINDOWS)
        self.cumulative += g
        self.run_min = np.minimum(self.run_min, self.cumulative)
        self.run_max = np.maximum(self.run_max, self.cumulative)
        for window in WINDOWS:
            self.window_sum[window] += g
            if self.t >= window:
                self.window_sum[window] -= self.ring[
                    (self.t - window) % max(WINDOWS)]

        out = np.empty(N_FEATURES, dtype=np.float64)
        out[:d3.N_FEATURES] = base
        col = d3.N_FEATURES
        sqrt_n = np.sqrt(n)
        for j in range(len(CHANNELS)):
            sd = self.sd[j]
            out[col] = self.cumulative[j] / (sd * sqrt_n)
            out[col + 1] = (self.cumulative[j] - self.run_min[j]) / (sd * sqrt_n)
            out[col + 2] = (self.run_max[j] - self.cumulative[j]) / (sd * sqrt_n)
            col += 3
            for window in WINDOWS:
                effective = min(n, window)
                out[col] = self.window_sum[window][j] / (sd * np.sqrt(effective))
                col += 1
        self.ring[pos] = g
        self.t += 1
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(11)
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
            post = online[tau:]
            online[tau:] = np.sign(post) * np.sqrt(np.abs(post))
        offline = compute_features_offline(hist, online)
        detector = StreamDetector11(hist)
        streamed = np.vstack([detector.step(v) for v in online])
        assert offline.shape == (T, N_FEATURES)
        error = float(np.max(np.abs(offline - streamed)))
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES
    detector = StreamDetector11(rng.normal(size=3000))
    values = rng.normal(size=1000)
    t0 = time.perf_counter()
    for value in values:
        detector.step(value)
    elapsed = (time.perf_counter() - t0) / len(values)
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
