"""Empirical-null adjacent-block contrasts for every proven d3 aspect.

Global history references can mistake a process's own slow variance or
amplitude evolution for a structural break.  This bank instead compares the
most recent block with the immediately preceding block at four scales.  Each
signed contrast is mapped through the distribution of the identical statistic
on that series' guaranteed break-free history.

At the boundary the preceding block is seeded from the historical tail, so
all features are available from the first online point.  Offline and streaming
paths are independent and tested for exact equality.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3
import detectors15 as d15


WINDOWS = (8, 32, 128, 256)
MAX_WINDOW = max(WINDOWS)
RING_SIZE = 2 * MAX_WINDOW
N_QUANTILES = 63
QUANTILE_PROBABILITIES = (
    np.arange(1, N_QUANTILES + 1, dtype=np.float64) /
    float(N_QUANTILES + 1))
N_ASPECTS = len(d3.ASPECTS)
N_PER_ASPECT = 2 * len(WINDOWS) + 2
N_FEATURES = N_ASPECTS * N_PER_ASPECT


def feature_names():
    names = []
    for aspect in d3.ASPECTS:
        names.extend(f"adj_{aspect}_w{window}" for window in WINDOWS)
        names.extend(f"absadj_{aspect}_w{window}" for window in WINDOWS)
        names.extend((f"adj_{aspect}_maxabs", f"adj_{aspect}_signedmax"))
    return names


def _contrasts(values, window, endpoints=None):
    """Adjacent recent-minus-previous block sums at 1-based endpoints."""
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    if endpoints is None:
        endpoints = np.arange(2 * int(window), values.size + 1,
                              dtype=np.int64)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    return (cumulative[endpoints]
            - 2.0 * cumulative[endpoints - int(window)]
            + cumulative[endpoints - 2 * int(window)])


def _scores(quantiles, values):
    values = np.round(np.asarray(values, dtype=np.float64), 8)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    left = np.sum(values[..., None] > quantiles, axis=-1)
    right = np.sum(values[..., None] >= quantiles, axis=-1)
    probability = (left + right + 1.0) / float(2 * (N_QUANTILES + 1))
    return ndtri(probability)


class AdjacentNull45:
    def __init__(self, x_hist):
        hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        self.hs = hs
        self.quantiles = np.empty(
            (N_ASPECTS, len(WINDOWS), N_QUANTILES), dtype=np.float64)
        self.tail = np.empty((N_ASPECTS, RING_SIZE), dtype=np.float64)
        for aspect_index, aspect in enumerate(d3.ASPECTS):
            values = (np.asarray(hist_inc[aspect], dtype=np.float64) /
                      hs.s[aspect])
            if values.size < RING_SIZE:
                raise ValueError("history is too short for adjacent contrasts")
            self.tail[aspect_index] = values[-RING_SIZE:]
            for window_index, window in enumerate(WINDOWS):
                contrast = _contrasts(values, int(window))
                self.quantiles[aspect_index, window_index] = np.round(
                    np.quantile(contrast, QUANTILE_PROBABILITIES), 8)

    def score_matrix(self, contrasts):
        return _scores(self.quantiles, contrasts)


def extra_features(x_hist, x_online):
    calibration = AdjacentNull45(x_hist)
    hs, _, online_inc = d15._hist_and_online_increments(x_hist, x_online)
    length = len(x_online)
    score = np.empty(
        (length, N_ASPECTS, len(WINDOWS)), dtype=np.float64)
    for aspect_index, aspect in enumerate(d3.ASPECTS):
        online = (np.asarray(online_inc[aspect], dtype=np.float64) /
                  hs.s[aspect])
        full = np.concatenate((calibration.tail[aspect_index], online))
        endpoints = RING_SIZE + np.arange(1, length + 1, dtype=np.int64)
        for window_index, window in enumerate(WINDOWS):
            contrast = _contrasts(full, int(window), endpoints=endpoints)
            score[:, aspect_index, window_index] = _scores(
                calibration.quantiles[aspect_index, window_index], contrast)

    output = np.empty((length, N_FEATURES), dtype=np.float64)
    for aspect_index in range(N_ASPECTS):
        column = aspect_index * N_PER_ASPECT
        block = score[:, aspect_index]
        output[:, column:column + len(WINDOWS)] = block
        output[:, column + len(WINDOWS):column + 2 * len(WINDOWS)] = (
            np.abs(block))
        index = np.argmax(np.abs(block), axis=1)
        maximum = block[np.arange(length), index]
        output[:, column + 2 * len(WINDOWS)] = np.abs(maximum)
        output[:, column + 2 * len(WINDOWS) + 1] = maximum
    return output


class StreamDetector45:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = AdjacentNull45(x_hist)
        self.ring = self.calibration.tail.copy()
        self.recent = np.empty(
            (N_ASPECTS, len(WINDOWS)), dtype=np.float64)
        self.previous = np.empty_like(self.recent)
        for window_index, window in enumerate(WINDOWS):
            self.recent[:, window_index] = self.ring[:, -int(window):].sum(
                axis=1)
            self.previous[:, window_index] = self.ring[
                :, -2 * int(window):-int(window)].sum(axis=1)
        self.t = 0

    def _from_current_inner(self):
        t = self.t
        position = t % RING_SIZE
        inner_position = (self.inner.t - 1) % self.inner._wmax
        current = np.asarray([
            self.inner._ring[aspect][inner_position] /
            self.inner.hs.s[aspect]
            for aspect in d3.ASPECTS
        ], dtype=np.float64)
        for window_index, window in enumerate(WINDOWS):
            lag_one = self.ring[
                :, (t + RING_SIZE - int(window)) % RING_SIZE]
            lag_two = self.ring[
                :, (t + RING_SIZE - 2 * int(window)) % RING_SIZE]
            self.recent[:, window_index] += current - lag_one
            self.previous[:, window_index] += lag_one - lag_two
        contrast = self.recent - self.previous
        score = self.calibration.score_matrix(contrast)

        output = np.empty(N_FEATURES, dtype=np.float64)
        for aspect_index in range(N_ASPECTS):
            column = aspect_index * N_PER_ASPECT
            block = score[aspect_index]
            output[column:column + len(WINDOWS)] = block
            output[column + len(WINDOWS):column + 2 * len(WINDOWS)] = (
                np.abs(block))
            index = int(np.argmax(np.abs(block)))
            maximum = float(block[index])
            output[column + 2 * len(WINDOWS)] = abs(maximum)
            output[column + 2 * len(WINDOWS) + 1] = maximum

        self.ring[:, position] = current
        self.t = t + 1
        return output

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(45)
    worst = 0.0
    for trial in range(10):
        n_history = int(rng.integers(1000, 3500))
        length = int(rng.integers(5, 800))
        total = n_history + length
        noise = rng.standard_t(rng.uniform(3.0, 12.0), total)
        values = np.empty(total, dtype=np.float64)
        values[0] = noise[0]
        coefficient = float(rng.uniform(-0.85, 0.85))
        for index in range(1, total):
            values[index] = coefficient * values[index - 1] + noise[index]
        history, online = values[:n_history], values[n_history:].copy()
        if trial % 2 == 0 and length > 30:
            tau = int(rng.integers(5, length - 5))
            online[tau:] += float(rng.uniform(-1.0, 1.0))
            online[tau:] *= float(rng.uniform(0.7, 1.6))

        offline = extra_features(history, online)
        detector = StreamDetector45(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        worst = max(worst, error)
        assert offline.shape == (length, N_FEATURES)
        assert error < 1e-9, (trial, error)

    detector = StreamDetector45(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_FEATURES
    print(f"OK: detectors45 stream == offline; {N_FEATURES} features; "
          f"max_error={worst:.3e}; {elapsed * 1e6:.0f} us/step")
