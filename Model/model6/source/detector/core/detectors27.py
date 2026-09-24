"""Empirical history-null ranks for the accepted d4x specialist aspects.

The two extreme-tail indicators and five multi-scale difference-energy
increments are calibrated by historical rolling-sum distributions at eight
causal horizons.  This is the d4x analogue of detectors22, restricted to the
seven aspects that already demonstrated complementary OOS value.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors4 as d4


ASPECTS = ("x99_up", "x99_dn", "d1e", "d2e", "d4e", "d8e", "absd1")
WINDOWS = (4, 8, 16, 32, 64, 128, 256, 512)
N_PER_ASPECT = len(WINDOWS) + 2
N_EXTRA = len(ASPECTS) * N_PER_ASPECT


def feature_names():
    names = []
    for aspect in ASPECTS:
        names.extend(f"emp4_{aspect}_w{window}" for window in WINDOWS)
        names.extend((f"emp4_{aspect}_maxabs", f"emp4_{aspect}_signedmax"))
    return names


def _normal_score(reference, values):
    values = np.round(np.asarray(values, dtype=np.float64), 8)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0)
    )
    return ndtri(probability)


class EmpiricalD4Null:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.hs = d4.Hist4(history)
        z = (history - self.hs.mu0) / self.hs.sd0
        u = self.hs._cdf(history)
        e = d4._ar_residuals(z, self.hs.phi, np.zeros(d4.AR_P))
        e = e[d4.AR_P:] if e.size > d4.AR_P else e
        if e.size < 10:
            e = z
        increments = d4._aspect_increments_hist(self.hs, z, u, e)
        self.reference = {}
        for aspect in ASPECTS:
            values = np.asarray(increments[aspect], dtype=np.float64)
            values = values / self.hs.s[aspect]
            cumulative = np.concatenate(([0.0], np.cumsum(values)))
            for window in WINDOWS:
                sums = cumulative[window:] - cumulative[:-window]
                self.reference[(aspect, window)] = np.sort(np.round(sums, 8))

    def score(self, aspect, window, values):
        return _normal_score(self.reference[(aspect, window)], values)

    def probability(self, aspect, window, value):
        reference = self.reference[(aspect, window)]
        rounded = round(float(value), 8)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))


def extra_features(x_hist, x_online):
    calibration = EmpiricalD4Null(x_hist)
    hs = calibration.hs
    values = np.asarray(x_online, dtype=np.float64)
    z = (values - hs.mu0) / hs.sd0
    u = hs._cdf(values)
    e = d4._ar_residuals(z, hs.phi, hs.zh_tail[-d4.AR_P:])
    increments = d4._aspect_increments_online(hs, z, u, e)
    steps = values.size
    output = np.zeros((steps, N_EXTRA), dtype=np.float64)
    column = 0
    for aspect in ASPECTS:
        g = np.asarray(increments[aspect], dtype=np.float64) / hs.s[aspect]
        cumulative = np.concatenate(([0.0], np.cumsum(g)))
        aspect_scores = []
        for window in WINDOWS:
            score = np.zeros(steps, dtype=np.float64)
            if steps >= window:
                sums = cumulative[window:] - cumulative[:-window]
                score[window - 1:] = calibration.score(aspect, window, sums)
            output[:, column] = score
            aspect_scores.append(score)
            column += 1
        matrix = np.column_stack(aspect_scores)
        index = np.argmax(np.abs(matrix), axis=1)
        output[:, column] = np.max(np.abs(matrix), axis=1)
        output[:, column + 1] = matrix[np.arange(steps), index]
        column += 2
    return output


class StreamDetector27:
    def __init__(self, x_hist, inner=None):
        self.inner = d4.StreamDetector4(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalD4Null(x_hist)
        self.w4_sum = {aspect: 0.0 for aspect in ASPECTS}

    def _from_current_inner(self):
        n = self.inner.t
        timestep = n - 1
        position = timestep % self.inner._wmax
        probabilities = np.full(
            (len(ASPECTS), len(WINDOWS)), 0.5, dtype=np.float64
        )
        for aspect_index, aspect in enumerate(ASPECTS):
            scale = self.inner.hs.s[aspect]
            current = self.inner._ring[aspect][position] / scale
            self.w4_sum[aspect] += current
            if timestep >= 4:
                self.w4_sum[aspect] -= (
                    self.inner._ring[aspect][
                        (timestep - 4) % self.inner._wmax
                    ] / scale
                )
            for window_index, window in enumerate(WINDOWS):
                if n >= window:
                    value = (
                        self.w4_sum[aspect] if window == 4
                        else self.inner._wsum[aspect][window] / scale
                    )
                    probabilities[aspect_index, window_index] = (
                        self.calibration.probability(aspect, window, value)
                    )
        score_matrix = ndtri(probabilities)
        output = np.empty(N_EXTRA, dtype=np.float64)
        for aspect_index in range(len(ASPECTS)):
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
    rng = np.random.default_rng(27)
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(10, 700))
        ar = rng.uniform(-0.85, 0.85)
        noise = rng.standard_t(rng.uniform(3.0, 12.0), n_hist + steps)
        values = np.empty(n_hist + steps, dtype=np.float64)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        history, online = values[:n_hist], values[n_hist:].copy()
        if trial % 2 == 0 and steps > 20:
            tau = int(rng.integers(5, steps - 5))
            online[tau:] *= rng.uniform(1.1, 1.8)
        offline = extra_features(history, online)
        detector = StreamDetector27(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d4.StreamDetector4(history)
        shared = StreamDetector27(history, inner=shared_inner)
        shared_rows = []
        for value in online:
            shared_inner.step(value)
            shared_rows.append(shared.step_from_inner())
        shared_streamed = np.vstack(shared_rows)
        error = float(np.max(np.abs(offline - streamed)))
        shared_error = float(np.max(np.abs(offline - shared_streamed)))
        assert offline.shape == (steps, N_EXTRA)
        assert error < 1e-9, (trial, error)
        assert shared_error < 1e-9, (trial, shared_error)
    detector = StreamDetector27(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_EXTRA
    print(
        f"OK: detectors27 stream == offline; {N_EXTRA} features; "
        f"{elapsed * 1e6:.0f} us/step"
    )
