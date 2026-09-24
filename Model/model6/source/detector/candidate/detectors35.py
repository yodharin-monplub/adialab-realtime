"""Causal random-Fourier MMD features on local lag embeddings.

The accepted banks describe marginal distributions and a modest collection
of serial moments.  This bank targets changes in the *joint distribution* of
short lag vectors.  Fixed random Fourier features approximate Gaussian
kernels in two per-series-normalized spaces:

* history-standardized values;
* history-ECDF normalised ranks.

Half the projections are level-sensitive and half use zero-sum weights to
isolate local shape.  Every projection is centered/scaled on break-free
history, then accumulated with causal prefix/CUSUM/suffix geometry.  Direct
grouped MMD energies are also standardized by their rolling history nulls.
"""

from __future__ import annotations

import time

import numpy as np


EMBED_LAGS = np.asarray((0, 1, 2, 4, 8, 16), dtype=np.int64)
MAX_LAG = int(EMBED_LAGS[-1])
RING_SIZE = MAX_LAG + 1
N_PROJECTIONS = 16
WINDOWS = (8, 32, 128, 512)
N_SCAN = 3 + len(WINDOWS)


def _make_projection_bank():
    rng = np.random.default_rng(350013)
    weights = rng.normal(size=(N_PROJECTIONS, EMBED_LAGS.size))
    channels = np.repeat(np.asarray((0, 1), dtype=np.int64),
                         N_PROJECTIONS // 2)
    shape_only = np.tile(
        np.repeat(np.asarray((0, 1), dtype=np.int64), 4), 2)
    for index in range(N_PROJECTIONS):
        if shape_only[index]:
            weights[index] -= weights[index].mean()
        norm = float(np.linalg.norm(weights[index]))
        weights[index] /= max(norm, 1e-12)
    scales = np.tile(np.asarray((0.5, 1.0, 2.0, 4.0)), 4)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=N_PROJECTIONS)
    groups = np.asarray([
        int(channels[index] * 2 + shape_only[index])
        for index in range(N_PROJECTIONS)
    ], dtype=np.int64)
    return weights, channels, shape_only, scales, phases, groups


WEIGHTS, CHANNELS, SHAPE_ONLY, SCALES, PHASES, GROUP_INDEX = (
    _make_projection_bank())
GROUPS = tuple(np.flatnonzero(GROUP_INDEX == group)
               for group in range(4))
N_GROUPS = len(GROUPS)
N_ENERGY = N_GROUPS * len(WINDOWS)
N_FEATURES = N_PROJECTIONS * N_SCAN + N_ENERGY


def feature_names():
    names = []
    for index in range(N_PROJECTIONS):
        channel = "z" if CHANNELS[index] == 0 else "rank"
        geometry = "shape" if SHAPE_ONLY[index] else "level"
        tag = (f"rff_{channel}_{geometry}_p{index:02d}_"
               f"s{SCALES[index]:g}")
        names.extend((f"{tag}_prefix", f"{tag}_up", f"{tag}_down"))
        names.extend(f"{tag}_w{window}" for window in WINDOWS)
    for group in range(N_GROUPS):
        channel = "z" if group // 2 == 0 else "rank"
        geometry = "level" if group % 2 == 0 else "shape"
        names.extend(f"rff_mmd_{channel}_{geometry}_w{window}"
                     for window in WINDOWS)
    return names


def _history_ranks(history):
    sorted_history = np.sort(history)
    left = np.searchsorted(sorted_history, history, side="left")
    right = np.searchsorted(sorted_history, history, side="right")
    return np.sqrt(12.0) * (
        (left + right) / (2.0 * history.size) - 0.5)


def _fixed_ranks(sorted_history, values):
    left = np.searchsorted(sorted_history, values, side="left")
    right = np.searchsorted(sorted_history, values, side="right")
    return np.sqrt(12.0) * (
        (left + right) / (2.0 * sorted_history.size) - 0.5)


def _responses(channels, positions):
    """RFF responses for absolute positions in a two-channel matrix."""
    positions = np.asarray(positions, dtype=np.int64)
    lag_positions = positions[:, None, None] - EMBED_LAGS[None, None, :]
    embedded = channels[CHANNELS[None, :, None], lag_positions]
    projected = np.sum(embedded * WEIGHTS[None, :, :], axis=2)
    return np.cos(projected * SCALES[None, :] + PHASES[None, :])


def _rolling_sums(values, window):
    cumulative = np.vstack((np.zeros((1, values.shape[1])),
                            np.cumsum(values, axis=0)))
    return cumulative[window:] - cumulative[:-window]


class Hist35:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        if history.size <= max(WINDOWS) + MAX_LAG:
            raise ValueError("RFF-MMD bank needs at least 529 history points")
        self.mu = float(history.mean())
        sd = float(history.std())
        self.sd = sd if sd > 1e-12 else 1e-12
        self.sorted_history = np.sort(history)
        z = (history - self.mu) / self.sd
        rank = _history_ranks(history)
        channels = np.vstack((z, rank))
        self.z_tail = z[-RING_SIZE:].copy()
        self.rank_tail = rank[-RING_SIZE:].copy()

        positions = np.arange(MAX_LAG, history.size, dtype=np.int64)
        raw = _responses(channels, positions)
        self.response_mean = raw.mean(axis=0)
        response_sd = raw.std(axis=0)
        self.response_sd = np.maximum(response_sd, 1e-9)
        increments = (raw - self.response_mean) / self.response_sd

        self.energy_mean = np.empty((N_GROUPS, len(WINDOWS)),
                                    dtype=np.float64)
        self.energy_sd = np.empty_like(self.energy_mean)
        # Rolling sums depend on the horizon, not on the projection group.
        # Compute each matrix once; the original group-first loop repeated
        # this relatively expensive history-null operation four times.
        rolling_by_window = [
            _rolling_sums(increments, int(window)) for window in WINDOWS
        ]
        for group, projection_indices in enumerate(GROUPS):
            for window_index, sums in enumerate(rolling_by_window):
                window = WINDOWS[window_index]
                energy = np.mean(
                    (sums[:, projection_indices] / np.sqrt(float(window))) ** 2,
                    axis=1)
                self.energy_mean[group, window_index] = float(energy.mean())
                scale = float(energy.std())
                self.energy_sd[group, window_index] = (
                    scale if scale > 1e-9 else 1.0)


def _online_increment_matrix(hist, x_online):
    online = np.asarray(x_online, dtype=np.float64)
    z = (online - hist.mu) / hist.sd
    rank = _fixed_ranks(hist.sorted_history, online)
    channels = np.vstack((np.concatenate((hist.z_tail, z)),
                          np.concatenate((hist.rank_tail, rank))))
    positions = RING_SIZE + np.arange(online.size, dtype=np.int64)
    raw = _responses(channels, positions)
    return (raw - hist.response_mean) / hist.response_sd


def compute_features_offline(x_hist, x_online):
    hist = Hist35(x_hist)
    increments = _online_increment_matrix(hist, x_online)
    length = increments.shape[0]
    if length == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    n = np.arange(1, length + 1, dtype=np.float64)
    cumulative = np.cumsum(increments, axis=0)
    padded = np.vstack((np.zeros((1, N_PROJECTIONS)), cumulative))
    running_min = np.minimum.accumulate(np.minimum(cumulative, 0.0), axis=0)
    running_max = np.maximum.accumulate(np.maximum(cumulative, 0.0), axis=0)

    output = np.empty((length, N_FEATURES), dtype=np.float64)
    scan = np.empty((length, N_PROJECTIONS, N_SCAN), dtype=np.float64)
    root_n = np.sqrt(n)[:, None]
    scan[:, :, 0] = cumulative / root_n
    scan[:, :, 1] = (cumulative - running_min) / root_n
    scan[:, :, 2] = (running_max - cumulative) / root_n
    index = np.arange(1, length + 1, dtype=np.int64)
    sums_by_window = []
    for column, window in enumerate(WINDOWS, start=3):
        start = np.maximum(0, index - int(window))
        sums = padded[index] - padded[start]
        sums_by_window.append(sums)
        scan[:, :, column] = sums / np.sqrt(
            np.minimum(n, float(window)))[:, None]
    output[:, :N_PROJECTIONS * N_SCAN] = scan.reshape(
        length, N_PROJECTIONS * N_SCAN)

    column = N_PROJECTIONS * N_SCAN
    for group, projection_indices in enumerate(GROUPS):
        for window_index, window in enumerate(WINDOWS):
            energy = np.zeros(length, dtype=np.float64)
            valid = n >= float(window)
            sums = sums_by_window[window_index]
            energy[valid] = np.mean(
                (sums[valid][:, projection_indices] /
                 np.sqrt(float(window))) ** 2,
                axis=1)
            energy[valid] = (
                (energy[valid] - hist.energy_mean[group, window_index]) /
                hist.energy_sd[group, window_index])
            output[:, column] = energy
            column += 1
    return output


class StreamDetector35:
    def __init__(self, x_hist):
        self.hist = Hist35(x_hist)
        self.values = np.vstack((self.hist.z_tail, self.hist.rank_tail))
        self.position = 0
        self.t = 0
        self.cumulative = np.zeros(N_PROJECTIONS, dtype=np.float64)
        self.minimum = np.zeros(N_PROJECTIONS, dtype=np.float64)
        self.maximum = np.zeros(N_PROJECTIONS, dtype=np.float64)
        self.ring = np.zeros((N_PROJECTIONS, max(WINDOWS)), dtype=np.float64)
        self.window_sum = np.zeros((N_PROJECTIONS, len(WINDOWS)),
                                   dtype=np.float64)

    def step(self, value):
        hist = self.hist
        raw_value = float(value)
        z = (raw_value - hist.mu) / hist.sd
        rank = float(_fixed_ranks(
            hist.sorted_history, np.asarray([raw_value]))[0])
        position = self.position
        self.values[0, position] = z
        self.values[1, position] = rank
        lag_positions = (position - EMBED_LAGS[None, :]) % RING_SIZE
        embedded = self.values[CHANNELS[:, None], lag_positions]
        projected = np.sum(embedded * WEIGHTS, axis=1)
        response = np.cos(projected * SCALES + PHASES)
        increment = (response - hist.response_mean) / hist.response_sd

        t = self.t
        n = t + 1
        self.cumulative += increment
        self.minimum = np.minimum(self.minimum, np.minimum(self.cumulative, 0.0))
        self.maximum = np.maximum(self.maximum, np.maximum(self.cumulative, 0.0))
        scan = np.empty((N_PROJECTIONS, N_SCAN), dtype=np.float64)
        root_n = np.sqrt(float(n))
        scan[:, 0] = self.cumulative / root_n
        scan[:, 1] = (self.cumulative - self.minimum) / root_n
        scan[:, 2] = (self.maximum - self.cumulative) / root_n

        ring_position = t % self.ring.shape[1]
        for column, window in enumerate(WINDOWS, start=3):
            window_index = column - 3
            self.window_sum[:, window_index] += increment
            if t >= int(window):
                self.window_sum[:, window_index] -= self.ring[
                    :, (t - int(window)) % self.ring.shape[1]]
            scan[:, column] = (
                self.window_sum[:, window_index] /
                np.sqrt(float(min(n, int(window)))))

        output = np.empty(N_FEATURES, dtype=np.float64)
        output[:N_PROJECTIONS * N_SCAN] = scan.reshape(
            N_PROJECTIONS * N_SCAN)
        column = N_PROJECTIONS * N_SCAN
        for group, projection_indices in enumerate(GROUPS):
            for window_index, window in enumerate(WINDOWS):
                if n < int(window):
                    output[column] = 0.0
                else:
                    sums = self.window_sum[projection_indices, window_index]
                    energy = float(np.mean(
                        (sums / np.sqrt(float(window))) ** 2))
                    output[column] = (
                        (energy - hist.energy_mean[group, window_index]) /
                        hist.energy_sd[group, window_index])
                column += 1

        self.ring[:, ring_position] = increment
        self.position = (position + 1) % RING_SIZE
        self.t = n
        return output


if __name__ == "__main__":
    rng = np.random.default_rng(20260813)
    worst = 0.0
    for trial in range(10):
        n_hist = int(rng.integers(700, 1800))
        length = int(rng.integers(5, 700))
        ar = float(rng.uniform(-0.8, 0.8))
        noise = rng.standard_t(float(rng.uniform(3.5, 15.0)),
                               size=n_hist + length)
        values = np.empty_like(noise)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        history = values[:n_hist].copy()
        online = values[n_hist:].copy()
        if trial % 2 == 0 and length > 30:
            tau = int(rng.integers(5, length - 5))
            for index in range(tau + 1, length):
                online[index] += 0.25 * np.tanh(online[index - 1])
        offline = compute_features_offline(history, online)
        detector = StreamDetector35(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        worst = max(worst, error)
        assert offline.shape == (length, N_FEATURES)
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES

    detector = StreamDetector35(rng.normal(size=3000))
    sample = rng.normal(size=1000)
    started = time.perf_counter()
    for value in sample:
        detector.step(value)
    elapsed = time.perf_counter() - started
    print(f"detectors35 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6 / len(sample):.0f} us/step")
