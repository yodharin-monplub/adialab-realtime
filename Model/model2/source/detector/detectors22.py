"""Empirical history-null quantiles for every aspect suffix statistic.

For each detectors3 aspect and fixed suffix length, historical rolling sums
form an empirical null distribution.  The current online suffix sum is mapped
to a signed normal score through that distribution.  This calibrates variance,
tail shape, and serial dependence simultaneously and separately per series.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3
import detectors15 as d15


WINDOWS = (4, 8, 16, 32, 64, 128, 256, 512)
N_ASPECTS = len(d3.ASPECTS)
N_PER_ASPECT = len(WINDOWS) + 2
N_EXTRA = N_ASPECTS * N_PER_ASPECT


def feature_names():
    names = []
    for aspect in d3.ASPECTS:
        names.extend(f"empq_{aspect}_w{window}" for window in WINDOWS)
        names.extend((f"empq_{aspect}_maxabs", f"empq_{aspect}_signedmax"))
    return names


def _normal_score(reference, value):
    # Rolling sums of discrete aspects have many exact ties.  Cumulative and
    # recurrent arithmetic can differ at 1e-15, which must not change a rank.
    reference = np.round(np.asarray(reference, dtype=np.float64), 8)
    values = np.round(np.asarray(value, dtype=np.float64), 8)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (2.0 * (reference.size + 1.0))
    return ndtri(probability)


class EmpiricalSuffixNull:
    def __init__(self, x_hist):
        hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        self.hs = hs
        self.reference = {}
        for aspect in d3.ASPECTS:
            g = np.asarray(hist_inc[aspect], dtype=np.float64) / hs.s[aspect]
            cumulative = np.concatenate(([0.0], np.cumsum(g)))
            for window in WINDOWS:
                sums = cumulative[window:] - cumulative[:-window]
                self.reference[(aspect, window)] = np.sort(np.round(sums, 8))

    def score(self, aspect, window, value):
        return _normal_score(self.reference[(aspect, window)], value)

    def probability(self, aspect, window, value):
        reference = self.reference[(aspect, window)]
        rounded = round(float(value), 8)
        # This scalar path runs for every aspect/window at every online step.
        # Calling the ndarray method avoids np.searchsorted's generic dispatch
        # wrapper while preserving the exact same binary-search semantics.
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))


def extra_features(x_hist, x_online):
    calibration = EmpiricalSuffixNull(x_hist)
    hs, _, increments = d15._hist_and_online_increments(x_hist, x_online)
    T = len(x_online)
    output = np.zeros((T, N_EXTRA), dtype=np.float64)
    column = 0
    for aspect in d3.ASPECTS:
        g = np.asarray(increments[aspect], dtype=np.float64) / hs.s[aspect]
        cumulative = np.concatenate(([0.0], np.cumsum(g)))
        aspect_scores = []
        for window in WINDOWS:
            z = np.zeros(T, dtype=np.float64)
            if T >= window:
                sums = cumulative[window:] - cumulative[:-window]
                z[window - 1:] = calibration.score(aspect, window, sums)
            output[:, column] = z
            aspect_scores.append(z)
            column += 1
        matrix = np.column_stack(aspect_scores)
        index = np.argmax(np.abs(matrix), axis=1)
        output[:, column] = np.max(np.abs(matrix), axis=1)
        output[:, column + 1] = matrix[np.arange(T), index]
        column += 2
    return output


class StreamDetector22:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalSuffixNull(x_hist)
        self.w4_sum = {aspect: 0.0 for aspect in d3.ASPECTS}

    def _from_current_inner(self):
        n = self.inner.t
        t = n - 1
        position = t % self.inner._wmax
        probabilities = np.full((N_ASPECTS, len(WINDOWS)), 0.5,
                                dtype=np.float64)
        for aspect_index, aspect in enumerate(d3.ASPECTS):
            scale = self.inner.hs.s[aspect]
            current = self.inner._ring[aspect][position] / scale
            self.w4_sum[aspect] += current
            if t >= 4:
                self.w4_sum[aspect] -= (
                    self.inner._ring[aspect][(t - 4) % self.inner._wmax] /
                    scale)
            for j, window in enumerate(WINDOWS):
                if n >= window:
                    value = (self.w4_sum[aspect] if window == 4 else
                             self.inner._wsum[aspect][window] / scale)
                    probabilities[aspect_index, j] = (
                        self.calibration.probability(
                            aspect, window, value))
        score_matrix = ndtri(probabilities)
        output = np.empty(N_EXTRA, dtype=np.float64)
        for aspect_index in range(N_ASPECTS):
            scores = score_matrix[aspect_index]
            column = aspect_index * N_PER_ASPECT
            output[column:column + len(WINDOWS)] = scores
            index = int(np.argmax(np.abs(scores)))
            output[column + len(WINDOWS)] = float(np.max(np.abs(scores)))
            output[column + len(WINDOWS) + 1] = scores[index]
        return output

    def step(self, x):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(x)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(22)
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        T = int(rng.integers(10, 700))
        ar = rng.uniform(-0.85, 0.85)
        eps = rng.standard_t(rng.uniform(3.0, 12.0), n_hist + T)
        values = np.empty(n_hist + T)
        values[0] = eps[0]
        for i in range(1, values.size):
            values[i] = ar * values[i - 1] + eps[i]
        hist, online = values[:n_hist], values[n_hist:].copy()
        if trial % 2 == 0 and T > 20:
            tau = int(rng.integers(5, T - 5))
            online[tau:] += rng.uniform(-1.0, 1.0)
            online[tau:] *= rng.uniform(1.1, 1.8)
        offline = extra_features(hist, online)
        stream = StreamDetector22(hist)
        streamed = np.vstack([stream.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (T, N_EXTRA)
        assert error < 1e-9, (trial, error)
    detector = StreamDetector22(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_EXTRA
    print(f"OK: detectors22 stream == offline; {N_EXTRA} features; "
          f"{elapsed*1e6:.0f} us/step")
