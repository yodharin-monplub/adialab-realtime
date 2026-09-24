"""Efficient score channels for autoregressive coefficient changes.

For a historical AR(5) fit, the Gaussian score for coefficient k is
proportional to e_t * z_{t-k}. A persistent coefficient change therefore
becomes a mean shift in these channels even when raw mean and variance stay
stable. Six lag scores are centered and scaled on break-free history, then
accumulated with the same causal CUSUM and suffix geometry as detectors3.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3


LAGS = (1, 2, 3, 5, 8, 13)
WINDOWS = d3.WINDOWS
BLOCK = 3 + len(WINDOWS)
N_FEATURES = len(LAGS) * BLOCK


def feature_names():
    names = []
    for lag in LAGS:
        prefix = f"arscore{lag}"
        names.extend((f"{prefix}_prefix", f"{prefix}_up",
                      f"{prefix}_down"))
        names.extend(f"{prefix}_w{window}" for window in WINDOWS)
    return names


class Hist24:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.base = d3.Hist3(history)
        z = (history - self.base.mu0) / self.base.sd0
        e = d3._ar_residuals(z, self.base.phi, np.zeros(d3.AR_P))
        self.mean = {}
        self.sd = {}
        for lag in LAGS:
            start = max(d3.AR_P, lag)
            score = e[start:] * z[start - lag:-lag]
            mean = float(score.mean()) if score.size else 0.0
            sd = float(score.std()) if score.size else 1.0
            self.mean[lag] = mean
            self.sd[lag] = sd if np.isfinite(sd) and sd > 1e-9 else 1.0
        tail = z[-max(LAGS):]
        if tail.size < max(LAGS):
            tail = np.concatenate([np.zeros(max(LAGS) - tail.size), tail])
        self.z_tail = tail


def _increments(hist, x_online):
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hist.base.mu0) / hist.base.sd0
    e = d3._ar_residuals(z, hist.base.phi, hist.base.zh_tail)
    full = np.concatenate([hist.z_tail, z])
    offset = hist.z_tail.size
    return {
        lag: e * full[offset - lag:offset - lag + z.size] - hist.mean[lag]
        for lag in LAGS
    }


def compute_features_offline(x_hist, x_online):
    hist = Hist24(x_hist)
    increments = _increments(hist, x_online)
    steps = len(x_online)
    n = np.arange(1, steps + 1, dtype=np.float64)
    sqrt_n = np.sqrt(n)
    index = np.arange(1, steps + 1)
    columns = []
    for lag in LAGS:
        score = increments[lag]
        scale = hist.sd[lag]
        cumulative = np.cumsum(score)
        padded = np.concatenate([[0.0], cumulative])
        run_min = np.minimum.accumulate(np.minimum(cumulative, 0.0))
        run_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        columns.append(cumulative / (scale * sqrt_n))
        columns.append((cumulative - run_min) / (scale * sqrt_n))
        columns.append((run_max - cumulative) / (scale * sqrt_n))
        for window in WINDOWS:
            start = np.maximum(0, index - window)
            effective = np.minimum(n, window)
            total = padded[1:] - padded[start]
            columns.append(total / (scale * np.sqrt(effective)))
    return np.column_stack(columns)


class StreamDetector24:
    def __init__(self, x_hist, inner=None):
        self.hist = Hist24(x_hist)
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.z_lags = list(self.hist.z_tail[::-1])
        self.t = 0
        self.cumulative = {lag: 0.0 for lag in LAGS}
        self.run_min = {lag: 0.0 for lag in LAGS}
        self.run_max = {lag: 0.0 for lag in LAGS}
        self.ring = {lag: np.zeros(max(WINDOWS)) for lag in LAGS}
        self.window_sum = {
            lag: {window: 0.0 for window in WINDOWS} for lag in LAGS
        }

    def _from_current_inner(self, x):
        z = (float(x) - self.hist.base.mu0) / self.hist.base.sd0
        e = self.inner.e_prev
        n = self.t + 1
        position = self.t % max(WINDOWS)
        output = np.empty(N_FEATURES, dtype=np.float64)
        column = 0
        for lag in LAGS:
            score = e * self.z_lags[lag - 1] - self.hist.mean[lag]
            scale = self.hist.sd[lag]
            cumulative = self.cumulative[lag] + score
            self.cumulative[lag] = cumulative
            self.run_min[lag] = min(self.run_min[lag], cumulative, 0.0)
            self.run_max[lag] = max(self.run_max[lag], cumulative, 0.0)
            root_n = np.sqrt(float(n))
            output[column] = cumulative / (scale * root_n)
            output[column + 1] = (cumulative - self.run_min[lag]) / (
                scale * root_n)
            output[column + 2] = (self.run_max[lag] - cumulative) / (
                scale * root_n)
            column += 3
            for window in WINDOWS:
                total = self.window_sum[lag][window] + score
                if self.t >= window:
                    total -= self.ring[lag][
                        (self.t - window) % max(WINDOWS)]
                self.window_sum[lag][window] = total
                effective = min(n, window)
                output[column] = total / (scale * np.sqrt(float(effective)))
                column += 1
            self.ring[lag][position] = score
        self.z_lags = [z] + self.z_lags[:-1]
        self.t = n
        return output

    def step(self, x):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(x)
        return self._from_current_inner(x)

    def step_from_inner(self, x):
        return self._from_current_inner(x)


if __name__ == "__main__":
    rng = np.random.default_rng(24)
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(10, 700))
        total = n_hist + steps
        change = n_hist + steps // 2
        values = np.zeros(total)
        noise = rng.normal(size=total)
        before = rng.uniform(-.7, .7)
        after = np.clip(before + rng.uniform(-.5, .5), -.9, .9)
        for index in range(1, total):
            coefficient = after if trial % 2 == 0 and index >= change else before
            values[index] = coefficient * values[index - 1] + noise[index]
        history, online = values[:n_hist], values[n_hist:]
        offline = compute_features_offline(history, online)
        detector = StreamDetector24(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (steps, N_FEATURES)
        assert error < 1e-9, (trial, error)
    detector = StreamDetector24(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_FEATURES
    print(f"OK: detectors24 stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
