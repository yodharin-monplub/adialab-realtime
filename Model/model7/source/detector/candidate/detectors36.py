"""Empirical-null multi-horizon local CUSUM scans.

The accepted d3 bank has global CUSUMs and fixed suffix sums.  This bank
fills the gap between them: at each horizon it scans every suffix ending at
the current point, so the candidate break age is not forced to coincide with
the start of that horizon.  Each upward/downward maximum is converted to an
empirical normal score using only the series' guaranteed break-free history.

The four horizons are deliberately sparse to control both feature count and
streaming cost.  The output contains eight calibrated scan scores plus the
strongest magnitude and its direction for every proven d3 aspect.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.special import ndtri

import detectors3 as d3
import detectors15 as d15


WINDOWS = (8, 32, 128, 512)
N_QUANTILES = 63
QUANTILE_PROBABILITIES = (
    np.arange(1, N_QUANTILES + 1, dtype=np.float64) /
    float(N_QUANTILES + 1))
N_ASPECTS = len(d3.ASPECTS)
N_PER_ASPECT = 2 * len(WINDOWS) + 2
N_FEATURES = N_ASPECTS * N_PER_ASPECT
RING_SIZE = max(WINDOWS) + 1


def feature_names():
    names = []
    for aspect in d3.ASPECTS:
        names.extend(f"empcus_{aspect}_up_w{window}" for window in WINDOWS)
        names.extend(f"empcus_{aspect}_dn_w{window}" for window in WINDOWS)
        names.extend((f"empcus_{aspect}_max", f"empcus_{aspect}_signedmax"))
    return names


def _rolling_extrema(values, window):
    """Min/max of every complete trailing window in compiled linear time."""
    values = np.asarray(values, dtype=np.float64)
    window = int(window)
    # origin=(window-1)//2 makes the ndimage filter causal: output i covers
    # [i-window+1, i].  We retain only complete windows, matching the former
    # sliding_window_view implementation exactly.
    origin = (window - 1) // 2
    minimum = minimum_filter1d(
        values, size=window, origin=origin,
        mode="constant", cval=np.inf)
    maximum = maximum_filter1d(
        values, size=window, origin=origin,
        mode="constant", cval=-np.inf)
    return minimum[window - 1:], maximum[window - 1:]


def _normal_score(reference, values):
    reference = np.asarray(reference, dtype=np.float64)
    rounded = np.round(np.asarray(values, dtype=np.float64), 8)
    left = np.searchsorted(reference, rounded, side="left")
    right = np.searchsorted(reference, rounded, side="right")
    probability = (left + right + 1.0) / (2.0 * (reference.size + 1.0))
    return ndtri(probability)


class EmpiricalLocalCusumNull:
    def __init__(self, x_hist):
        hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        self.hs = hs
        self.up_quantiles = np.empty(
            (N_ASPECTS, len(WINDOWS), N_QUANTILES), dtype=np.float64)
        self.down_quantiles = np.empty_like(self.up_quantiles)
        for aspect_index, aspect in enumerate(d3.ASPECTS):
            increment = (np.asarray(hist_inc[aspect], dtype=np.float64) /
                         hs.s[aspect])
            cumulative = np.concatenate(([0.0], np.cumsum(increment)))
            for window_index, window in enumerate(WINDOWS):
                minimum, maximum = _rolling_extrema(
                    cumulative, int(window) + 1)
                endpoint = cumulative[int(window):]
                up = endpoint - minimum
                down = maximum - endpoint
                self.up_quantiles[aspect_index, window_index] = np.round(
                    np.quantile(up, QUANTILE_PROBABILITIES), 8)
                self.down_quantiles[aspect_index, window_index] = np.round(
                    np.quantile(down, QUANTILE_PROBABILITIES), 8)

    def score(self, aspect, window, up, down):
        aspect_index = d3.ASPECTS.index(aspect)
        window_index = WINDOWS.index(window)
        return (
            _normal_score(
                self.up_quantiles[aspect_index, window_index], up),
            _normal_score(
                self.down_quantiles[aspect_index, window_index], down),
        )

    @staticmethod
    def _probability(reference, value):
        rounded = round(float(value), 8)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))

    def matrix_probabilities(self, up, down, active):
        """Vectorized 64-bin empirical lookup over aspect and horizon."""
        up = np.round(np.asarray(up, dtype=np.float64), 8)
        down = np.round(np.asarray(down, dtype=np.float64), 8)
        probabilities_up = np.full(
            (N_ASPECTS, len(WINDOWS)), 0.5, dtype=np.float64)
        probabilities_down = np.full_like(probabilities_up, 0.5)
        active = np.asarray(active, dtype=bool)
        if np.any(active):
            up_left = np.sum(
                up[:, active, None] > self.up_quantiles[:, active, :],
                axis=2)
            up_right = np.sum(
                up[:, active, None] >= self.up_quantiles[:, active, :],
                axis=2)
            down_left = np.sum(
                down[:, active, None] > self.down_quantiles[:, active, :],
                axis=2)
            down_right = np.sum(
                down[:, active, None] >= self.down_quantiles[:, active, :],
                axis=2)
            denominator = float(2 * (N_QUANTILES + 1))
            probabilities_up[:, active] = (
                up_left + up_right + 1.0) / denominator
            probabilities_down[:, active] = (
                down_left + down_right + 1.0) / denominator
        return probabilities_up, probabilities_down


def extra_features(x_hist, x_online):
    calibration = EmpiricalLocalCusumNull(x_hist)
    hs, _, increments = d15._hist_and_online_increments(x_hist, x_online)
    length = len(x_online)
    output = np.zeros((length, N_FEATURES), dtype=np.float64)
    for aspect_index, aspect in enumerate(d3.ASPECTS):
        increment = (np.asarray(increments[aspect], dtype=np.float64) /
                     hs.s[aspect])
        cumulative = np.concatenate(([0.0], np.cumsum(increment)))
        up_scores = np.zeros((length, len(WINDOWS)), dtype=np.float64)
        down_scores = np.zeros_like(up_scores)
        for window_index, window in enumerate(WINDOWS):
            if length < int(window):
                continue
            minimum, maximum = _rolling_extrema(
                cumulative, int(window) + 1)
            endpoint = cumulative[int(window):]
            up, down = calibration.score(
                aspect, window, endpoint - minimum, maximum - endpoint)
            up_scores[int(window) - 1:, window_index] = up
            down_scores[int(window) - 1:, window_index] = down

        column = aspect_index * N_PER_ASPECT
        output[:, column:column + len(WINDOWS)] = up_scores
        output[:, column + len(WINDOWS):
               column + 2 * len(WINDOWS)] = down_scores
        combined = np.concatenate((up_scores, down_scores), axis=1)
        index = np.argmax(combined, axis=1)
        maximum = combined[np.arange(length), index]
        direction = np.where(index < len(WINDOWS), 1.0, -1.0)
        output[:, column + 2 * len(WINDOWS)] = maximum
        output[:, column + 2 * len(WINDOWS) + 1] = maximum * direction
    return output


class StreamDetector36:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalLocalCusumNull(x_hist)
        self.cumulative = np.zeros(N_ASPECTS, dtype=np.float64)
        self.min_deques = [
            [deque(((0, 0.0),)) for _ in range(len(WINDOWS))]
            for _ in range(N_ASPECTS)
        ]
        self.max_deques = [
            [deque(((0, 0.0),)) for _ in range(len(WINDOWS))]
            for _ in range(N_ASPECTS)
        ]

    def _from_current_inner(self):
        n = self.inner.t
        t = n - 1
        increment_position = t % self.inner._wmax
        increment = np.asarray([
            self.inner._ring[aspect][increment_position] /
            self.inner.hs.s[aspect]
            for aspect in d3.ASPECTS
        ], dtype=np.float64)
        self.cumulative += increment

        up_matrix = np.zeros((N_ASPECTS, len(WINDOWS)), dtype=np.float64)
        down_matrix = np.zeros_like(up_matrix)
        active = np.zeros(len(WINDOWS), dtype=bool)
        for window_index, window in enumerate(WINDOWS):
            lower = n - int(window)
            for aspect_index in range(N_ASPECTS):
                value = float(self.cumulative[aspect_index])
                minimum = self.min_deques[aspect_index][window_index]
                maximum = self.max_deques[aspect_index][window_index]
                while minimum and minimum[-1][1] >= value:
                    minimum.pop()
                minimum.append((n, value))
                while maximum and maximum[-1][1] <= value:
                    maximum.pop()
                maximum.append((n, value))
                while minimum[0][0] < lower:
                    minimum.popleft()
                while maximum[0][0] < lower:
                    maximum.popleft()
            if n < int(window):
                continue
            active[window_index] = True
            up_matrix[:, window_index] = np.asarray([
                self.cumulative[aspect_index] -
                self.min_deques[aspect_index][window_index][0][1]
                for aspect_index in range(N_ASPECTS)
            ])
            down_matrix[:, window_index] = np.asarray([
                self.max_deques[aspect_index][window_index][0][1] -
                self.cumulative[aspect_index]
                for aspect_index in range(N_ASPECTS)
            ])
        probabilities_up, probabilities_down = (
            self.calibration.matrix_probabilities(
                up_matrix, down_matrix, active))
        probabilities = np.concatenate(
            (probabilities_up, probabilities_down), axis=1)
        scores = ndtri(probabilities)
        output = np.empty(N_FEATURES, dtype=np.float64)
        for aspect_index in range(N_ASPECTS):
            column = aspect_index * N_PER_ASPECT
            row = scores[aspect_index]
            output[column:column + 2 * len(WINDOWS)] = row
            index = int(np.argmax(row))
            maximum = float(row[index])
            output[column + 2 * len(WINDOWS)] = maximum
            output[column + 2 * len(WINDOWS) + 1] = (
                maximum if index < len(WINDOWS) else -maximum)
        return output

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(20260813)
    worst = 0.0
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        length = int(rng.integers(10, 700))
        ar = float(rng.uniform(-0.85, 0.85))
        noise = rng.standard_t(float(rng.uniform(3.0, 12.0)),
                               n_hist + length)
        values = np.empty_like(noise)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        history = values[:n_hist].copy()
        online = values[n_hist:].copy()
        if trial % 2 == 0 and length > 30:
            tau = int(rng.integers(5, length - 5))
            online[tau:] += float(rng.uniform(-0.8, 0.8))
            online[tau:] *= float(rng.uniform(0.7, 1.5))
        offline = extra_features(history, online)
        detector = StreamDetector36(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        worst = max(worst, error)
        assert offline.shape == (length, N_FEATURES)
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES

    detector = StreamDetector36(rng.normal(size=3000))
    sample = rng.normal(size=1000)
    started = time.perf_counter()
    for value in sample:
        detector.step(value)
    elapsed = time.perf_counter() - started
    print(f"detectors36 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6 / len(sample):.0f} us/step")
