"""Empirical-null Haar contrasts of the historical ECDF.

The detectors3 distribution channels are global low-order moments. This bank
adds localized, signed probability-mass contrasts: for every dyadic interval
through level four, +1 denotes its lower half, -1 its upper half, and 0 lies
outside the interval. Each contrast is implicitly centered on the guaranteed
break-free history. Suffix sums are mapped to empirical normal scores using rolling sums
from that same history, calibrating tails and serial dependence per series.

Only the online prefix ending at the current observation is used. The offline
and streaming paths implement the same discrete empirical-rank calculation.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3


LEVELS = tuple(range(5))
CHANNELS = tuple((level, cell)
                 for level in LEVELS for cell in range(1 << level))
WINDOWS = (4, 8, 16, 32, 64, 128, 256, 512)
N_CHANNELS = len(CHANNELS)
N_PER_CHANNEL = len(WINDOWS) + 2
N_FEATURES = N_CHANNELS * N_PER_CHANNEL


def feature_names():
    names = []
    for level, cell in CHANNELS:
        prefix = f"haar_l{level}_c{cell}"
        names.extend(f"{prefix}_w{window}" for window in WINDOWS)
        names.extend((f"{prefix}_maxabs", f"{prefix}_signedmax"))
    return names


def _haar_values(u):
    values = np.asarray(u, dtype=np.float64).reshape(-1)
    # An online value beyond the historical maximum has ECDF rank exactly 1.
    # Assign it to the final dyadic cell rather than outside [0, 1).
    values = np.clip(values, 0.0, np.nextafter(1.0, 0.0))
    output = np.zeros((values.size, N_CHANNELS), dtype=np.int8)
    for column, (level, cell) in enumerate(CHANNELS):
        scale = 1 << level
        local = values * scale - cell
        active = (local >= 0.0) & (local < 1.0)
        output[active & (local < 0.5), column] = 1.0
        output[active & (local >= 0.5), column] = -1.0
    return output


class Hist23:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.base = d3.Hist3(history)
        raw = _haar_values(self.base._cdf(history))
        cumulative = np.vstack([
            np.zeros((1, N_CHANNELS), dtype=np.int32),
            np.cumsum(raw, axis=0, dtype=np.int32),
        ])
        self.lookup = {}
        for window in WINDOWS:
            sums = cumulative[window:] - cumulative[:-window]
            possible = np.arange(-window, window + 1, dtype=np.int32)
            table = np.empty((N_CHANNELS, possible.size), dtype=np.float64)
            for channel in range(N_CHANNELS):
                reference = np.sort(sums[:, channel])
                left = np.searchsorted(reference, possible, side="left")
                right = np.searchsorted(reference, possible, side="right")
                probability = (left + right + 1.0) / (
                    2.0 * (reference.size + 1.0))
                table[channel] = ndtri(probability)
            self.lookup[window] = table

    def raw_values(self, values):
        x = np.asarray(values, dtype=np.float64)
        return _haar_values(self.base._cdf(x))


def compute_features_offline(x_hist, x_online):
    hist = Hist23(x_hist)
    increments = hist.raw_values(x_online)
    steps = increments.shape[0]
    cumulative = np.vstack([
        np.zeros((1, N_CHANNELS), dtype=np.int32),
        np.cumsum(increments, axis=0, dtype=np.int32),
    ])
    output = np.zeros((steps, N_FEATURES), dtype=np.float64)
    for channel in range(N_CHANNELS):
        column = channel * N_PER_CHANNEL
        scores = np.zeros((steps, len(WINDOWS)), dtype=np.float64)
        for offset, window in enumerate(WINDOWS):
            if steps >= window:
                sums = cumulative[window:, channel] - cumulative[
                    :-window, channel]
                scores[window - 1:, offset] = hist.lookup[window][
                    channel, sums + window]
            output[:, column + offset] = scores[:, offset]
        index = np.argmax(np.abs(scores), axis=1)
        output[:, column + len(WINDOWS)] = np.max(
            np.abs(scores), axis=1)
        output[:, column + len(WINDOWS) + 1] = scores[
            np.arange(steps), index]
    return output


class StreamDetector23:
    def __init__(self, x_hist):
        self.hist = Hist23(x_hist)
        self.t = 0
        self.ring = np.zeros((max(WINDOWS), N_CHANNELS), dtype=np.int8)
        self.window_sum = {
            window: np.zeros(N_CHANNELS, dtype=np.int32)
            for window in WINDOWS
        }

    def step(self, x):
        increment = self.hist.raw_values([float(x)])[0]
        position = self.t % max(WINDOWS)
        n = self.t + 1
        score_matrix = np.zeros((N_CHANNELS, len(WINDOWS)),
                                dtype=np.float64)
        for offset, window in enumerate(WINDOWS):
            current = self.window_sum[window]
            current += increment
            if self.t >= window:
                current -= self.ring[(self.t - window) % max(WINDOWS)]
            if n >= window:
                score_matrix[:, offset] = self.hist.lookup[window][
                    np.arange(N_CHANNELS), current + window]
        output = np.empty(N_FEATURES, dtype=np.float64)
        for channel in range(N_CHANNELS):
            scores = score_matrix[channel]
            column = channel * N_PER_CHANNEL
            output[column:column + len(WINDOWS)] = scores
            index = int(np.argmax(np.abs(scores)))
            output[column + len(WINDOWS)] = float(np.max(np.abs(scores)))
            output[column + len(WINDOWS) + 1] = scores[index]
        self.ring[position] = increment
        self.t = n
        return output


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(10, 700))
        ar = rng.uniform(-0.85, 0.85)
        noise = rng.standard_t(rng.uniform(3.0, 12.0), n_hist + steps)
        values = np.empty(n_hist + steps)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        if trial == 0:  # exercise tied empirical ranks as well
            values = np.round(values, 1)
        history, online = values[:n_hist], values[n_hist:].copy()
        if trial % 2 == 0 and steps > 20:
            tau = int(rng.integers(5, steps - 5))
            post = online[tau:]
            online[tau:] = np.sign(post) * np.sqrt(np.abs(post))
        offline = compute_features_offline(history, online)
        detector = StreamDetector23(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (steps, N_FEATURES)
        assert error < 1e-9, (trial, error)
    detector = StreamDetector23(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_FEATURES
    print(f"OK: detectors23 stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
