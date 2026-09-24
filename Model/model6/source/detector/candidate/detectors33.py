"""Deterministic causal MiniROCKET-style local-pattern detector bank.

Thirty-two fixed zero-sum length-9 kernels are spread over dilations 1..32
and two complementary input spaces: history-standardized values and fixed
historical ECDF ranks.  Each kernel bias is a deterministic quantile of its
own break-free historical responses.  The online increment is the change in
the kernel's positive-proportion indicator relative to that history.

This is a compact streaming analogue of ROCKET PPV features.  It detects
changes in local motifs without assuming that the old and current challenge
distributions match.  Prefix/CUSUM and four suffix geometries are normalized
by each kernel's own historical Bernoulli null standard deviation.
"""

from __future__ import annotations

import itertools
import time

import numpy as np


N_KERNELS = 16
KERNEL_LENGTH = 9
DILATION_CHOICES = np.asarray((1, 2, 4, 8), dtype=np.int64)
WINDOWS = (8, 32, 128, 512)
N_PER_KERNEL = 3 + len(WINDOWS)
N_FEATURES = N_KERNELS * N_PER_KERNEL


def _make_kernels():
    combinations = list(itertools.combinations(range(KERNEL_LENGTH), 3))
    weights = np.full((N_KERNELS, KERNEL_LENGTH), -1.0, dtype=np.float64)
    for index in range(N_KERNELS):
        combo = combinations[(index * 37 + 11) % len(combinations)]
        weights[index, list(combo)] = 2.0
    weights /= np.sqrt(np.sum(weights * weights, axis=1, keepdims=True))
    # Match the public EvoForest graph excerpt: four kernels at each of
    # dilations 1/2/4/8.  Alternate value/rank space within each dilation.
    dilations = np.repeat(DILATION_CHOICES, 4)
    channels = np.tile(np.asarray((0, 1, 0, 1), dtype=np.int64), 4)
    quantile_choices = np.asarray((0.10, 0.25, 0.50, 0.75, 0.90))
    quantiles = quantile_choices[(np.arange(N_KERNELS) * 3 + 2) %
                                 quantile_choices.size]
    return weights, dilations, channels, quantiles


KERNELS, DILATIONS, CHANNELS, BIAS_QUANTILES = _make_kernels()
SPANS = (KERNEL_LENGTH - 1) * DILATIONS
MAX_SPAN = int(SPANS.max())
RING_SIZE = MAX_SPAN + 1
LAGS = (np.arange(KERNEL_LENGTH, dtype=np.int64)[None, :] *
        DILATIONS[:, None])
KERNEL_GROUPS = tuple(
    (int(channel), int(dilation),
     np.flatnonzero((CHANNELS == channel) & (DILATIONS == dilation)))
    for channel in (0, 1)
    for dilation in DILATION_CHOICES
    if np.any((CHANNELS == channel) & (DILATIONS == dilation))
)


def feature_names():
    names = []
    for index in range(N_KERNELS):
        channel = "z" if CHANNELS[index] == 0 else "rank"
        tag = f"rocket_{channel}_k{index:02d}_d{int(DILATIONS[index])}"
        names.extend((f"{tag}_prefix", f"{tag}_up", f"{tag}_down"))
        names.extend(f"{tag}_w{window}" for window in WINDOWS)
    return names


def _history_ranks(history):
    order = np.argsort(history, kind="mergesort")
    sorted_history = history[order]
    left = np.searchsorted(sorted_history, history, side="left")
    right = np.searchsorted(sorted_history, history, side="right")
    return (left + right) / (2.0 * history.size) - 0.5


def _fixed_ranks(sorted_history, values):
    left = np.searchsorted(sorted_history, values, side="left")
    right = np.searchsorted(sorted_history, values, side="right")
    return (left + right) / (2.0 * sorted_history.size) - 0.5


def _online_responses(hist, x_online):
    online = np.asarray(x_online, dtype=np.float64)
    z = (online - hist.mu) / hist.sd
    rank = _fixed_ranks(hist.sorted_history, online)
    channels = np.vstack((np.concatenate((hist.z_tail, z)),
                          np.concatenate((hist.rank_tail, rank))))
    length = online.size
    positions = (RING_SIZE + np.arange(length, dtype=np.int64)[:, None, None]
                 - LAGS[None, :, :])
    gathered = channels[CHANNELS[None, :, None], positions]
    return np.sum(gathered * KERNELS[None, :, :], axis=2)


class Hist33:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        if history.size <= MAX_SPAN:
            raise ValueError(f"rocket bank needs more than {MAX_SPAN} history points")
        self.mu = float(history.mean())
        sd = float(history.std())
        self.sd = sd if sd > 1e-12 else 1e-12
        self.sorted_history = np.sort(history)
        z = (history - self.mu) / self.sd
        rank = _history_ranks(history)
        self.z_tail = z[-RING_SIZE:].copy()
        self.rank_tail = rank[-RING_SIZE:].copy()

        self.bias = np.empty(N_KERNELS, dtype=np.float64)
        self.null_rate = np.empty(N_KERNELS, dtype=np.float64)
        self.null_sd = np.empty(N_KERNELS, dtype=np.float64)
        channels = np.vstack((z, rank))
        positions = (np.arange(MAX_SPAN, history.size, dtype=np.int64)
                     [:, None, None] - LAGS[None, :, :])
        gathered = channels[CHANNELS[None, :, None], positions]
        responses = np.sum(
            gathered * KERNELS[None, :, :], axis=2)
        for index in range(N_KERNELS):
            current = responses[:, index]
            bias = float(np.quantile(
                current, float(BIAS_QUANTILES[index]), method="linear"))
            indicator = (current > bias).astype(np.float64)
            rate = float(indicator.mean())
            self.bias[index] = bias
            self.null_rate[index] = rate
            scale = float(indicator.std())
            self.null_sd[index] = scale if scale > 1e-9 else 1e-9


def _online_increment_matrix(hist, x_online):
    response = _online_responses(hist, x_online)
    return ((np.round(response, 12) > hist.bias[None, :]).astype(np.float64) -
            hist.null_rate[None, :])


def compute_features_offline(x_hist, x_online):
    hist = Hist33(x_hist)
    increments = _online_increment_matrix(hist, x_online)
    length = increments.shape[0]
    if length == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    n = np.arange(1, length + 1, dtype=np.float64)
    cumulative = np.cumsum(increments, axis=0)
    running_min = np.minimum.accumulate(np.minimum(cumulative, 0.0), axis=0)
    running_max = np.maximum.accumulate(np.maximum(cumulative, 0.0), axis=0)
    padded = np.vstack((np.zeros((1, N_KERNELS)), cumulative))
    output = np.empty((length, N_KERNELS, N_PER_KERNEL), dtype=np.float64)
    denominator = hist.null_sd[None, :] * np.sqrt(n)[:, None]
    output[:, :, 0] = cumulative / denominator
    output[:, :, 1] = (cumulative - running_min) / denominator
    output[:, :, 2] = (running_max - cumulative) / denominator
    index = np.arange(1, length + 1, dtype=np.int64)
    for column, window in enumerate(WINDOWS, start=3):
        start = np.maximum(0, index - int(window))
        total = padded[index] - padded[start]
        effective = np.minimum(n, float(window))
        output[:, :, column] = (
            total / (hist.null_sd[None, :] * np.sqrt(effective)[:, None]))
    return output.reshape(length, N_FEATURES)


class StreamDetector33:
    def __init__(self, x_hist):
        self.hist = Hist33(x_hist)
        self.values = np.vstack((self.hist.z_tail, self.hist.rank_tail))
        self.position = 0
        self.t = 0
        self.cumulative = np.zeros(N_KERNELS, dtype=np.float64)
        self.minimum = np.zeros(N_KERNELS, dtype=np.float64)
        self.maximum = np.zeros(N_KERNELS, dtype=np.float64)
        self.ring = np.zeros((N_KERNELS, max(WINDOWS)), dtype=np.float64)
        self.window_sum = np.zeros((N_KERNELS, len(WINDOWS)),
                                   dtype=np.float64)

    def step(self, value):
        hist = self.hist
        raw = float(value)
        z = (raw - hist.mu) / hist.sd
        rank = float(_fixed_ranks(hist.sorted_history, np.asarray([raw]))[0])
        position = self.position
        self.values[0, position] = z
        self.values[1, position] = rank

        indices = (position - LAGS) % RING_SIZE
        gathered = self.values[CHANNELS[:, None], indices]
        response = np.sum(KERNELS * gathered, axis=1)
        increment = ((np.round(response, 12) > hist.bias).astype(np.float64) -
                     hist.null_rate)

        t = self.t
        n = t + 1
        self.cumulative += increment
        self.minimum = np.minimum(self.minimum, np.minimum(self.cumulative, 0.0))
        self.maximum = np.maximum(self.maximum, np.maximum(self.cumulative, 0.0))

        output = np.empty((N_KERNELS, N_PER_KERNEL), dtype=np.float64)
        denominator = hist.null_sd * np.sqrt(float(n))
        output[:, 0] = self.cumulative / denominator
        output[:, 1] = (self.cumulative - self.minimum) / denominator
        output[:, 2] = (self.maximum - self.cumulative) / denominator
        ring_size = self.ring.shape[1]
        ring_position = t % ring_size
        for column, window in enumerate(WINDOWS, start=3):
            window_index = column - 3
            self.window_sum[:, window_index] += increment
            drop = t - int(window)
            if drop >= 0:
                self.window_sum[:, window_index] -= self.ring[:, drop % ring_size]
            output[:, column] = (
                self.window_sum[:, window_index] /
                (hist.null_sd * np.sqrt(float(min(n, int(window))))))
        self.ring[:, ring_position] = increment

        self.position = (position + 1) % RING_SIZE
        self.t = n
        return output.reshape(N_FEATURES)


if __name__ == "__main__":
    rng = np.random.default_rng(20260813)
    worst = 0.0
    for trial in range(12):
        n_hist = int(rng.integers(500, 1400))
        length = int(rng.integers(5, 750))
        ar = float(rng.uniform(-0.8, 0.8))
        noise = rng.standard_t(df=float(rng.uniform(3.5, 15.0)),
                               size=n_hist + length)
        series = np.empty_like(noise)
        series[0] = noise[0]
        for index in range(1, series.size):
            series[index] = ar * series[index - 1] + noise[index]
        history = series[:n_hist].copy()
        online = series[n_hist:].copy()
        if trial % 3 == 0 and length > 30:
            tau = int(rng.integers(5, length - 5))
            for index in range(tau + 1, length):
                online[index] = 0.75 * online[index - 1] + noise[n_hist + index]
        elif trial % 3 == 1 and length > 30:
            tau = int(rng.integers(5, length - 5))
            online[tau:] *= 1.6

        offline = compute_features_offline(history, online)
        detector = StreamDetector33(history)
        streaming = np.vstack([detector.step(x) for x in online])
        error = float(np.max(np.abs(offline - streaming)))
        worst = max(worst, error)
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES

    detector = StreamDetector33(rng.normal(size=3000))
    sample = rng.normal(size=1000)
    started = time.perf_counter()
    for value in sample:
        detector.step(value)
    elapsed = time.perf_counter() - started
    print(f"detectors33 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6 / len(sample):.0f} us/step")
