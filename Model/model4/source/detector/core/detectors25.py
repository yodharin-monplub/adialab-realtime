"""History-calibrated nonlinear serial-copula change features.

The historical empirical CDF maps each observation to a per-series rank in
[0, 1].  Products of low-order shifted Legendre modes at several lags measure
nonlinear serial dependence beyond ordinary autocorrelation.  Every product
is centered and scaled on the break-free history, then accumulated with the
same causal prefix/CUSUM/suffix geometry used by detectors3.

Both paths deliberately use historical lag context for the first online
observations.  No online or future observation enters the calibration.
"""

from __future__ import annotations

import time

import numpy as np


LAGS = (1, 2, 5, 13)
# (current Legendre degree, lagged Legendre degree)
MOMENTS = ((1, 1), (2, 2), (3, 3), (1, 2), (2, 1))
WINDOWS = (8, 16, 32, 64, 128, 256, 512)
BLOCK = 3 + len(WINDOWS)
N_CHANNELS = len(LAGS) * len(MOMENTS)
N_FEATURES = N_CHANNELS * BLOCK


def feature_names():
    names = []
    for lag in LAGS:
        for current_degree, lag_degree in MOMENTS:
            prefix = f"cop_l{lag}_p{current_degree}{lag_degree}"
            names.extend((f"{prefix}_prefix", f"{prefix}_up",
                          f"{prefix}_down"))
            names.extend(f"{prefix}_w{window}" for window in WINDOWS)
    return names


def _legendre(u):
    """Return shifted Legendre modes P1..P3 as rows."""
    u = np.asarray(u, dtype=np.float64)
    return np.vstack((
        2.0 * u - 1.0,
        6.0 * u * u - 6.0 * u + 1.0,
        20.0 * u ** 3 - 30.0 * u * u + 12.0 * u - 1.0,
    ))


class Hist25:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.n_hist = history.size
        self.sorted_hist = np.sort(history)
        ranks = self.cdf(history)
        modes = _legendre(ranks)
        self.mean = {}
        self.sd = {}
        for lag in LAGS:
            for current_degree, lag_degree in MOMENTS:
                values = (
                    modes[current_degree - 1, lag:]
                    * modes[lag_degree - 1, :-lag]
                )
                mean = float(values.mean())
                sd = float(values.std())
                key = (lag, current_degree, lag_degree)
                self.mean[key] = mean
                self.sd[key] = sd if np.isfinite(sd) and sd > 1e-9 else 1e-9
        max_lag = max(LAGS)
        tail = modes[:, -max_lag:]
        if tail.shape[1] < max_lag:
            tail = np.column_stack((
                np.zeros((3, max_lag - tail.shape[1])), tail
            ))
        self.mode_tail = tail

    def cdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        lo = np.searchsorted(self.sorted_hist, x, side="left")
        hi = np.searchsorted(self.sorted_hist, x, side="right")
        return (lo + hi) / (2.0 * self.n_hist)


def _increments(hist, x_online):
    ranks = hist.cdf(np.asarray(x_online, dtype=np.float64))
    modes = _legendre(ranks)
    full = np.column_stack((hist.mode_tail, modes))
    offset = hist.mode_tail.shape[1]
    increments = []
    for lag in LAGS:
        for current_degree, lag_degree in MOMENTS:
            key = (lag, current_degree, lag_degree)
            lagged = full[
                lag_degree - 1, offset - lag:offset - lag + modes.shape[1]
            ]
            values = modes[current_degree - 1] * lagged - hist.mean[key]
            increments.append(values)
    return increments


def compute_features_offline(x_hist, x_online):
    hist = Hist25(x_hist)
    increments = _increments(hist, x_online)
    steps = len(x_online)
    n = np.arange(1, steps + 1, dtype=np.float64)
    root_n = np.sqrt(n)
    index = np.arange(1, steps + 1)
    columns = []
    for channel, values in enumerate(increments):
        lag = LAGS[channel // len(MOMENTS)]
        current_degree, lag_degree = MOMENTS[channel % len(MOMENTS)]
        scale = hist.sd[(lag, current_degree, lag_degree)]
        cumulative = np.cumsum(values)
        padded = np.concatenate(([0.0], cumulative))
        run_min = np.minimum.accumulate(np.minimum(cumulative, 0.0))
        run_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        columns.append(cumulative / (scale * root_n))
        columns.append((cumulative - run_min) / (scale * root_n))
        columns.append((run_max - cumulative) / (scale * root_n))
        for window in WINDOWS:
            start = np.maximum(0, index - window)
            effective = np.minimum(n, window)
            total = padded[1:] - padded[start]
            columns.append(total / (scale * np.sqrt(effective)))
    return np.column_stack(columns)


class StreamDetector25:
    def __init__(self, x_hist):
        self.hist = Hist25(x_hist)
        # Most recent mode vector first.
        self.mode_lags = [
            self.hist.mode_tail[:, -(lag + 1)]
            for lag in range(max(LAGS))
        ]
        self.t = 0
        self.cumulative = np.zeros(N_CHANNELS, dtype=np.float64)
        self.run_min = np.zeros(N_CHANNELS, dtype=np.float64)
        self.run_max = np.zeros(N_CHANNELS, dtype=np.float64)
        self.ring = np.zeros((N_CHANNELS, max(WINDOWS)), dtype=np.float64)
        self.window_sum = np.zeros((N_CHANNELS, len(WINDOWS)), dtype=np.float64)

    def step(self, x):
        rank = float(self.hist.cdf(np.asarray([x]))[0])
        current = _legendre(np.asarray([rank]))[:, 0]
        n = self.t + 1
        position = self.t % max(WINDOWS)
        root_n = np.sqrt(float(n))
        output = np.empty(N_FEATURES, dtype=np.float64)
        channel = 0
        column = 0
        for lag in LAGS:
            lagged = self.mode_lags[lag - 1]
            for current_degree, lag_degree in MOMENTS:
                key = (lag, current_degree, lag_degree)
                value = (
                    current[current_degree - 1] * lagged[lag_degree - 1]
                    - self.hist.mean[key]
                )
                scale = self.hist.sd[key]
                cumulative = self.cumulative[channel] + value
                self.cumulative[channel] = cumulative
                self.run_min[channel] = min(
                    self.run_min[channel], cumulative, 0.0
                )
                self.run_max[channel] = max(
                    self.run_max[channel], cumulative, 0.0
                )
                output[column] = cumulative / (scale * root_n)
                output[column + 1] = (
                    cumulative - self.run_min[channel]
                ) / (scale * root_n)
                output[column + 2] = (
                    self.run_max[channel] - cumulative
                ) / (scale * root_n)
                column += 3
                for window_index, window in enumerate(WINDOWS):
                    total = self.window_sum[channel, window_index] + value
                    if self.t >= window:
                        total -= self.ring[
                            channel, (self.t - window) % max(WINDOWS)
                        ]
                    self.window_sum[channel, window_index] = total
                    effective = min(n, window)
                    output[column] = total / (
                        scale * np.sqrt(float(effective))
                    )
                    column += 1
                self.ring[channel, position] = value
                channel += 1
        self.mode_lags = [current] + self.mode_lags[:-1]
        self.t = n
        return output


if __name__ == "__main__":
    rng = np.random.default_rng(25)
    for trial in range(12):
        n_hist = int(rng.integers(1000, 3500))
        steps = int(rng.integers(10, 800))
        total = n_hist + steps
        values = np.zeros(total, dtype=np.float64)
        noise = rng.normal(size=total)
        coefficient = rng.uniform(-0.75, 0.75)
        for index in range(1, total):
            values[index] = coefficient * values[index - 1] + noise[index]
        if trial % 2 == 0:
            change = n_hist + steps // 2
            # Nonlinear post-break dependence with little forced mean shift.
            for index in range(change, total):
                previous = np.clip(values[index - 1], -3.0, 3.0)
                values[index] += 0.25 * (previous ** 2 - 1.0)
        history, online = values[:n_hist], values[n_hist:]
        offline = compute_features_offline(history, online)
        detector = StreamDetector25(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (steps, N_FEATURES)
        assert error < 1e-9, (trial, error)
    detector = StreamDetector25(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_FEATURES
    print(
        f"OK: detectors25 stream == offline; {N_FEATURES} features; "
        f"{elapsed * 1e6:.0f} us/step"
    )
