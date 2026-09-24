"""Long-lag difference-energy specialist with exact streaming parity.

Extends the successful detectors4 energy mechanism from lags 1/2/4/8 to
16/32/64/128.  For every lag k it tracks both squared and absolute
standardized differences relative to their break-free historical means.
Each increment is accumulated as prefix/CUSUM and suffix-window statistics
normalized by its own historical null standard deviation.
"""

from __future__ import annotations

import time

import numpy as np


WINDOWS = (8, 16, 32, 64, 128, 256, 512)
LAGS = (16, 32, 64, 128)
MAX_LAG = max(LAGS)
ASPECTS = tuple([f"d{k}e" for k in LAGS] +
                [f"absd{k}" for k in LAGS])
N_FEATURES = len(ASPECTS) * (3 + len(WINDOWS))


def feature_names():
    names = []
    for aspect in ASPECTS:
        names += [f"{aspect}_prefix", f"{aspect}_up", f"{aspect}_down"]
        names += [f"{aspect}_w{w}" for w in WINDOWS]
    return names


class Hist12:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.mu = float(h.mean())
        sd = float(h.std())
        self.sd = sd if sd > 1e-12 else 1e-12
        z = (h - self.mu) / self.sd
        self.tail = (z[-MAX_LAG:] if z.size >= MAX_LAG else
                     np.concatenate([np.zeros(MAX_LAG - z.size), z]))
        self.energy_mean = {}
        self.abs_mean = {}
        increments = {}
        for k in LAGS:
            if z.size > k:
                diff = z[k:] - z[:-k]
                energy_mean = float(np.mean(diff * diff))
                abs_mean = float(np.mean(np.abs(diff)))
            else:
                energy_mean, abs_mean = 2.0, 1.0
            self.energy_mean[k] = energy_mean
            self.abs_mean[k] = abs_mean
            ge = np.zeros(z.size, dtype=np.float64)
            ga = np.zeros(z.size, dtype=np.float64)
            if z.size > k:
                diff = z[k:] - z[:-k]
                ge[k:] = diff * diff - energy_mean
                ga[k:] = np.abs(diff) - abs_mean
            increments[f"d{k}e"] = ge
            increments[f"absd{k}"] = ga
        self.null_sd = {}
        for aspect in ASPECTS:
            value = float(np.std(increments[aspect]))
            self.null_sd[aspect] = value if value > 1e-9 else 1e-9


def _online_increments(hist, z):
    out = {}
    for k in LAGS:
        full = np.concatenate([hist.tail[-k:], z])
        diff = z - full[:z.size]
        out[f"d{k}e"] = diff * diff - hist.energy_mean[k]
        out[f"absd{k}"] = np.abs(diff) - hist.abs_mean[k]
    return out


def compute_features_offline(x_hist, x_online):
    hist = Hist12(x_hist)
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hist.mu) / hist.sd
    increments = _online_increments(hist, z)
    n = np.arange(1, z.size + 1, dtype=np.float64)
    sqrt_n = np.sqrt(n)
    idx = np.arange(1, z.size + 1)
    starts = {w: np.maximum(0, idx - w) for w in WINDOWS}
    effective = {w: np.minimum(n, w) for w in WINDOWS}
    columns = []
    for aspect in ASPECTS:
        g = increments[aspect]
        scale = hist.null_sd[aspect]
        cumulative = np.cumsum(g)
        padded = np.concatenate([[0.0], cumulative])
        run_min = np.minimum.accumulate(np.minimum(cumulative, 0.0))
        run_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        columns.append(cumulative / (scale * sqrt_n))
        columns.append((cumulative - run_min) / (scale * sqrt_n))
        columns.append((run_max - cumulative) / (scale * sqrt_n))
        for window in WINDOWS:
            total = padded[1:] - padded[starts[window]]
            columns.append(total / (scale * np.sqrt(effective[window])))
    return np.column_stack(columns)


class StreamDetector12:
    def __init__(self, x_hist):
        self.hist = Hist12(x_hist)
        self.t = 0
        self.lags = list(self.hist.tail[::-1])
        self.cumulative = {a: 0.0 for a in ASPECTS}
        self.minimum = {a: 0.0 for a in ASPECTS}
        self.maximum = {a: 0.0 for a in ASPECTS}
        self.ring = {a: np.zeros(max(WINDOWS)) for a in ASPECTS}
        self.window_sum = {
            a: {w: 0.0 for w in WINDOWS} for a in ASPECTS
        }

    def step(self, value):
        hist = self.hist
        z = (float(value) - hist.mu) / hist.sd
        increment = {}
        for k in LAGS:
            diff = z - self.lags[k - 1]
            increment[f"d{k}e"] = diff * diff - hist.energy_mean[k]
            increment[f"absd{k}"] = abs(diff) - hist.abs_mean[k]

        t = self.t
        n = t + 1
        sqrt_n = np.sqrt(float(n))
        ring_size = max(WINDOWS)
        position = t % ring_size
        out = np.empty(N_FEATURES, dtype=np.float64)
        column = 0
        for aspect in ASPECTS:
            g = increment[aspect]
            scale = hist.null_sd[aspect]
            cumulative = self.cumulative[aspect] + g
            self.cumulative[aspect] = cumulative
            self.minimum[aspect] = min(self.minimum[aspect], cumulative, 0.0)
            self.maximum[aspect] = max(self.maximum[aspect], cumulative, 0.0)
            out[column] = cumulative / (scale * sqrt_n)
            out[column + 1] = ((cumulative - self.minimum[aspect]) /
                               (scale * sqrt_n))
            out[column + 2] = ((self.maximum[aspect] - cumulative) /
                               (scale * sqrt_n))
            column += 3
            ring = self.ring[aspect]
            sums = self.window_sum[aspect]
            for window in WINDOWS:
                sums[window] += g
                drop = t - window
                if drop >= 0:
                    sums[window] -= ring[drop % ring_size]
                out[column] = (sums[window] /
                               (scale * np.sqrt(float(min(n, window)))))
                column += 1
            ring[position] = g

        self.lags = [z] + self.lags[:MAX_LAG - 1]
        self.t = n
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(1201)
    for trial in range(12):
        n_hist = int(rng.integers(180, 700))
        n_online = int(rng.integers(5, 850))
        ar = rng.uniform(-0.75, 0.75)
        noise = rng.normal(size=n_hist + n_online)
        x = np.empty(noise.size)
        x[0] = noise[0]
        for i in range(1, x.size):
            x[i] = ar * x[i - 1] + noise[i]
        hist = x[:n_hist].copy()
        online = x[n_hist:].copy()
        if trial % 2 == 0 and n_online > 20:
            tau = int(rng.integers(1, n_online - 1))
            if trial % 4 == 0:
                online[tau:] += rng.uniform(0.4, 1.4)
            else:
                online[tau:] *= rng.uniform(1.3, 2.1)
        offline = compute_features_offline(hist, online)
        detector = StreamDetector12(hist)
        streamed = np.vstack([detector.step(v) for v in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (n_online, N_FEATURES)
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES

    detector = StreamDetector12(rng.normal(size=3000))
    online = rng.normal(size=1000)
    start = time.perf_counter()
    for value in online:
        detector.step(value)
    elapsed = (time.perf_counter() - start) / online.size
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
