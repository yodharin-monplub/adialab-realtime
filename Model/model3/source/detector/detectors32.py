"""Causal Haar-wavelet band-energy features with history null calibration.

This bank targets changes in the *frequency allocation* of a series, a gap
left by the point-lag difference energies in :mod:`detectors4`.  At every
online point it computes normalized Haar coefficients whose two adjacent
half-windows have lengths 2..128.  Squared coefficients are divided by their
own break-free historical means, making the resulting evidence comparable
across series.  Four cheap spectral-shape summaries describe how the current
energy is distributed across scales.

Every aspect is accumulated with the same prefix/CUSUM/suffix geometry as the
accepted detector banks.  The offline path is vectorized and the streaming
path uses O(number of scales + number of features) work per point.
"""

from __future__ import annotations

import time

import numpy as np


HALF_WINDOWS = np.asarray((2, 4, 8, 16, 32, 64, 128), dtype=np.int64)
WINDOWS = (8, 16, 32, 64, 128, 256, 512)
RING_SIZE = int(2 * HALF_WINDOWS[-1])
COMMON_START = RING_SIZE - 1
SCALE_COORD = np.linspace(-1.0, 1.0, HALF_WINDOWS.size)
SCALE_DENOM = float(np.dot(SCALE_COORD, SCALE_COORD))

BAND_ASPECTS = tuple(f"haar{int(h)}e" for h in HALF_WINDOWS)
SPECTRAL_ASPECTS = (
    "wave_entropy",
    "wave_centroid",
    "wave_slope",
    "wave_peak",
)
ASPECTS = BAND_ASPECTS + SPECTRAL_ASPECTS
N_PER_ASPECT = 3 + len(WINDOWS)
N_FEATURES = len(ASPECTS) * N_PER_ASPECT


def feature_names():
    names = []
    for aspect in ASPECTS:
        names.extend((f"{aspect}_prefix", f"{aspect}_up",
                      f"{aspect}_down"))
        names.extend(f"{aspect}_w{window}" for window in WINDOWS)
    return names


def _haar_matrix(values, positions):
    """Return normalized adjacent-block Haar coefficients at ``positions``."""
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    columns = []
    for half in HALF_WINDOWS:
        half = int(half)
        recent = (cumulative[positions + 1] -
                  cumulative[positions - half + 1])
        previous = (cumulative[positions - half + 1] -
                    cumulative[positions - 2 * half + 1])
        columns.append((recent - previous) / np.sqrt(2.0 * half))
    return np.column_stack(columns)


def _spectral_summaries(relative_energy):
    """Cheap entropy/centroid/slope/peak summaries across Haar bands."""
    q = np.asarray(relative_energy, dtype=np.float64)
    safe = np.maximum(q, 1e-12)
    probability = safe / np.maximum(safe.sum(axis=1, keepdims=True), 1e-12)
    entropy = (-np.sum(probability * np.log(probability), axis=1) /
               np.log(float(q.shape[1])))
    centroid = probability @ SCALE_COORD
    slope = (np.log(safe + 0.05) @ SCALE_COORD) / SCALE_DENOM
    peak = np.max(probability, axis=1)
    return np.column_stack((entropy, centroid, slope, peak))


class Hist32:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        if history.size <= COMMON_START:
            raise ValueError(
                f"wavelet bank needs at least {COMMON_START + 1} history points")
        self.mu = float(history.mean())
        sd = float(history.std())
        self.sd = sd if sd > 1e-12 else 1e-12
        z = (history - self.mu) / self.sd
        self.tail = z[-RING_SIZE:].copy()

        positions = np.arange(COMMON_START, z.size, dtype=np.int64)
        coefficient = _haar_matrix(z, positions)
        energy = coefficient * coefficient
        self.energy_mean = np.maximum(energy.mean(axis=0), 1e-12)
        relative = energy / self.energy_mean
        spectral = _spectral_summaries(relative)
        self.spectral_mean = spectral.mean(axis=0)

        increments = np.column_stack(
            (relative - 1.0, spectral - self.spectral_mean))
        scale = increments.std(axis=0)
        self.null_sd = np.maximum(scale, 1e-9)


def _online_increment_matrix(hist, x_online):
    online = np.asarray(x_online, dtype=np.float64)
    z = (online - hist.mu) / hist.sd
    full = np.concatenate((hist.tail, z))
    positions = RING_SIZE + np.arange(z.size, dtype=np.int64)
    coefficient = _haar_matrix(full, positions)
    relative = coefficient * coefficient / hist.energy_mean
    spectral = _spectral_summaries(relative)
    return np.column_stack(
        (relative - 1.0, spectral - hist.spectral_mean))


def compute_features_offline(x_hist, x_online):
    hist = Hist32(x_hist)
    increments = _online_increment_matrix(hist, x_online)
    length = increments.shape[0]
    if length == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)

    n = np.arange(1, length + 1, dtype=np.float64)
    cumulative = np.cumsum(increments, axis=0)
    running_min = np.minimum.accumulate(np.minimum(cumulative, 0.0), axis=0)
    running_max = np.maximum.accumulate(np.maximum(cumulative, 0.0), axis=0)
    padded = np.vstack((np.zeros((1, len(ASPECTS))), cumulative))

    values = np.empty((length, len(ASPECTS), N_PER_ASPECT),
                      dtype=np.float64)
    denominator = hist.null_sd[None, :] * np.sqrt(n)[:, None]
    values[:, :, 0] = cumulative / denominator
    values[:, :, 1] = (cumulative - running_min) / denominator
    values[:, :, 2] = (running_max - cumulative) / denominator

    index = np.arange(1, length + 1, dtype=np.int64)
    for column, window in enumerate(WINDOWS, start=3):
        start = np.maximum(0, index - int(window))
        total = padded[index] - padded[start]
        effective = np.minimum(n, float(window))
        values[:, :, column] = (
            total / (hist.null_sd[None, :] * np.sqrt(effective)[:, None]))
    return values.reshape(length, N_FEATURES)


class StreamDetector32:
    def __init__(self, x_hist):
        self.hist = Hist32(x_hist)
        self.t = 0
        self.position = 0
        self.values = self.hist.tail.copy()

        self.recent = np.asarray([
            self.values[-int(half):].sum() for half in HALF_WINDOWS
        ], dtype=np.float64)
        self.previous = np.asarray([
            self.values[-2 * int(half):-int(half)].sum()
            for half in HALF_WINDOWS
        ], dtype=np.float64)

        n_aspects = len(ASPECTS)
        self.cumulative = np.zeros(n_aspects, dtype=np.float64)
        self.minimum = np.zeros(n_aspects, dtype=np.float64)
        self.maximum = np.zeros(n_aspects, dtype=np.float64)
        self.ring = np.zeros((n_aspects, max(WINDOWS)), dtype=np.float64)
        self.window_sum = np.zeros((n_aspects, len(WINDOWS)),
                                   dtype=np.float64)

    def step(self, value):
        hist = self.hist
        z = (float(value) - hist.mu) / hist.sd
        position = self.position

        for index, half_value in enumerate(HALF_WINDOWS):
            half = int(half_value)
            lag_half = self.values[(position + RING_SIZE - half) % RING_SIZE]
            lag_full = self.values[(position + RING_SIZE - 2 * half) % RING_SIZE]
            self.recent[index] += z - lag_half
            self.previous[index] += lag_half - lag_full

        coefficient = ((self.recent - self.previous) /
                       np.sqrt(2.0 * HALF_WINDOWS))
        relative = coefficient * coefficient / hist.energy_mean
        spectral = _spectral_summaries(relative[None, :])[0]
        increment = np.concatenate(
            (relative - 1.0, spectral - hist.spectral_mean))

        t = self.t
        n = t + 1
        sqrt_n = np.sqrt(float(n))
        self.cumulative += increment
        self.minimum = np.minimum(self.minimum, np.minimum(self.cumulative, 0.0))
        self.maximum = np.maximum(self.maximum, np.maximum(self.cumulative, 0.0))

        output = np.empty((len(ASPECTS), N_PER_ASPECT), dtype=np.float64)
        denominator = hist.null_sd * sqrt_n
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

        self.values[position] = z
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
        noise = rng.normal(size=n_hist + length)
        series = np.empty_like(noise)
        series[0] = noise[0]
        for index in range(1, series.size):
            series[index] = ar * series[index - 1] + noise[index]
        history = series[:n_hist].copy()
        online = series[n_hist:].copy()
        if trial % 3 == 0 and length > 30:
            tau = int(rng.integers(5, length - 5))
            online[tau:] += 0.7 * np.sin(
                np.arange(length - tau, dtype=np.float64) * np.pi / 4.0)
        elif trial % 3 == 1 and length > 30:
            tau = int(rng.integers(5, length - 5))
            online[tau:] += 0.5

        offline = compute_features_offline(history, online)
        detector = StreamDetector32(history)
        streaming = np.vstack([detector.step(x) for x in online])
        error = float(np.max(np.abs(offline - streaming)))
        worst = max(worst, error)
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES

    detector = StreamDetector32(rng.normal(size=3000))
    sample = rng.normal(size=1000)
    started = time.perf_counter()
    for value in sample:
        detector.step(value)
    elapsed = time.perf_counter() - started
    print(f"detectors32 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6 / len(sample):.0f} us/step")
