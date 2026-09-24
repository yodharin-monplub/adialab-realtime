"""Empirical-null complete coarse serial-copula transition bank.

The endpoint generator microscope found transferable signal in changes to a
4x4 historical-quantile transition matrix.  A complete 3x3 tensor basis of
integer Haar contrasts spans the dependence part of that matrix while using
only nine channels rather than sixteen sparse cells.  This detector applies
that basis at lags 1/2/4/8 and maps every causal suffix sum through its own
rolling break-free-history null distribution.  Because every increment is
in {-1, 0, 1}, all null scores are precomputed in small integer lookup tables.

Unlike detectors25, this bank includes all nine cross-mode combinations,
uses the diagnostic lags exactly, and empirically calibrates every horizon.
The offline and streaming paths are independent implementations of the same
math and are tested for exact agreement below.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri


LAGS = (1, 2, 4, 8)
DEGREES = (1, 2, 3)
WINDOWS = (8, 16, 32, 64, 128, 256, 512)
N_CHANNELS = len(LAGS) * len(DEGREES) ** 2
N_PER_CHANNEL = len(WINDOWS) + 2
N_FEATURES = N_CHANNELS * N_PER_CHANNEL

# Three independent centered contrasts of four historical-quantile states.
# Their tensor products span all 3*3 dependence degrees of freedom.
CONTRASTS = np.asarray((
    (-1, -1, 1, 1),
    (-1, 1, 0, 0),
    (0, 0, -1, 1),
), dtype=np.int8)


def feature_names():
    names = []
    for lag in LAGS:
        for current_degree in DEGREES:
            for lagged_degree in DEGREES:
                prefix = (f"qtrans_l{lag}_h{current_degree}"
                          f"{lagged_degree}")
                names.extend(f"{prefix}_w{window}" for window in WINDOWS)
                names.extend((f"{prefix}_maxabs", f"{prefix}_signedmax"))
    return names


def _normal_score(reference, values):
    values = np.asarray(values)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0))
    return ndtri(probability)


class Hist40:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.edges = np.quantile(history, (0.25, 0.50, 0.75))
        state = np.searchsorted(self.edges, history, side="right")
        modes = CONTRASTS[:, state]
        self.mode_tail = modes[:, -max(LAGS):].copy()
        self.lookup = [
            np.empty((N_CHANNELS, 2 * window + 1), dtype=np.float64)
            for window in WINDOWS
        ]

        channel = 0
        for lag in LAGS:
            for current_degree in DEGREES:
                for lagged_degree in DEGREES:
                    values = (modes[current_degree - 1, lag:] *
                              modes[lagged_degree - 1, :-lag])
                    cumulative = np.concatenate((
                        np.zeros(1, dtype=np.int64),
                        np.cumsum(values, dtype=np.int64)))
                    for window_index, window in enumerate(WINDOWS):
                        sums = cumulative[window:] - cumulative[:-window]
                        reference = np.sort(sums)
                        possible = np.arange(-window, window + 1,
                                             dtype=np.int64)
                        self.lookup[window_index][channel] = _normal_score(
                            reference, possible)
                    channel += 1

    def modes(self, values):
        state = np.searchsorted(
            self.edges, np.asarray(values, dtype=np.float64), side="right")
        return CONTRASTS[:, state]


def _online_increments(hist, x_online):
    modes = hist.modes(x_online)
    full = np.column_stack((hist.mode_tail, modes))
    offset = hist.mode_tail.shape[1]
    increments = []
    for lag in LAGS:
        for current_degree in DEGREES:
            for lagged_degree in DEGREES:
                lagged = full[
                    lagged_degree - 1,
                    offset - lag:offset - lag + modes.shape[1],
                ]
                increments.append(modes[current_degree - 1] * lagged)
    return np.column_stack(increments).astype(np.int8, copy=False)


def compute_features_offline(x_hist, x_online):
    hist = Hist40(x_hist)
    increments = _online_increments(hist, x_online)
    steps = increments.shape[0]
    scores = np.zeros((steps, N_CHANNELS, len(WINDOWS)), dtype=np.float64)
    cumulative = np.vstack((np.zeros((1, N_CHANNELS), dtype=np.int64),
                            np.cumsum(increments, axis=0, dtype=np.int64)))
    for window_index, window in enumerate(WINDOWS):
        if steps < window:
            continue
        sums = cumulative[window:] - cumulative[:-window]
        lookup = hist.lookup[window_index]
        scores[window - 1:, :, window_index] = lookup[
            np.arange(N_CHANNELS)[None, :], sums + window]

    output = np.empty((steps, N_FEATURES), dtype=np.float64)
    for channel in range(N_CHANNELS):
        column = channel * N_PER_CHANNEL
        block = scores[:, channel, :]
        output[:, column:column + len(WINDOWS)] = block
        index = np.argmax(np.abs(block), axis=1)
        output[:, column + len(WINDOWS)] = np.max(np.abs(block), axis=1)
        output[:, column + len(WINDOWS) + 1] = block[
            np.arange(steps), index]
    return output


class StreamDetector40:
    def __init__(self, x_hist):
        self.hist = Hist40(x_hist)
        # Most recent historical mode vector first.
        self.mode_lags = [
            self.hist.mode_tail[:, -(offset + 1)].copy()
            for offset in range(max(LAGS))
        ]
        self.t = 0
        self.ring = np.zeros((N_CHANNELS, max(WINDOWS)), dtype=np.int8)
        self.window_sum = np.zeros(
            (N_CHANNELS, len(WINDOWS)), dtype=np.int64)

    def step(self, x):
        current = self.hist.modes(np.asarray([x]))[:, 0]
        values = np.empty(N_CHANNELS, dtype=np.int8)
        channel = 0
        for lag in LAGS:
            outer = current[:, None] * self.mode_lags[lag - 1][None, :]
            values[channel:channel + 9] = outer.reshape(-1)
            channel += 9

        n = self.t + 1
        position = self.t % max(WINDOWS)
        score = np.zeros(
            (N_CHANNELS, len(WINDOWS)), dtype=np.float64)
        for window_index, window in enumerate(WINDOWS):
            sums = self.window_sum[:, window_index] + values
            if self.t >= window:
                sums -= self.ring[:, (self.t - window) % max(WINDOWS)]
            self.window_sum[:, window_index] = sums
            if n >= window:
                score[:, window_index] = self.hist.lookup[window_index][
                    np.arange(N_CHANNELS), sums + window]
        output = np.empty(N_FEATURES, dtype=np.float64)
        for channel in range(N_CHANNELS):
            column = channel * N_PER_CHANNEL
            block = score[channel]
            output[column:column + len(WINDOWS)] = block
            index = int(np.argmax(np.abs(block)))
            output[column + len(WINDOWS)] = float(np.max(np.abs(block)))
            output[column + len(WINDOWS) + 1] = block[index]

        self.ring[:, position] = values
        self.mode_lags = [current] + self.mode_lags[:-1]
        self.t = n
        return output


if __name__ == "__main__":
    rng = np.random.default_rng(40)
    worst = 0.0
    for trial in range(10):
        n_history = int(rng.integers(1000, 3500))
        steps = int(rng.integers(10, 800))
        total = n_history + steps
        noise = rng.standard_t(rng.uniform(3.0, 12.0), total)
        values = np.empty(total, dtype=np.float64)
        values[0] = noise[0]
        coefficient = rng.uniform(-0.8, 0.8)
        for index in range(1, total):
            values[index] = coefficient * values[index - 1] + noise[index]
        history, online = values[:n_history], values[n_history:].copy()
        if trial % 2 == 0 and steps > 30:
            tau = int(rng.integers(8, steps - 8))
            # Inject a nonlinear sign-dependent transition change.
            for index in range(tau + 1, steps):
                online[index] += 0.55 * np.sign(online[index - 1])

        offline = compute_features_offline(history, online)
        detector = StreamDetector40(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        worst = max(worst, error)
        assert offline.shape == (steps, N_FEATURES)
        assert error < 1e-9, (trial, error)

    detector = StreamDetector40(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_FEATURES
    print(f"OK: detectors40 stream == offline; {N_FEATURES} features; "
          f"max_error={worst:.3e}; {elapsed * 1e6:.0f} us/step")
