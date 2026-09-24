"""Empirical history-null calibration of detectors15 EWMA evidence.

For every detectors3 aspect and EWMA decay, the break-free history supplies
the empirical distribution of the recursively filtered state.  The causal
online state is finite-start variance corrected, mapped through that empirical
CDF, and emitted as a signed normal score plus its absolute value.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3
import detectors15 as d15
import detectors22 as d22


ALPHA_INDICES = np.asarray((0, 2, 4, 6), dtype=np.int64)
ALPHAS = d15.ALPHAS[ALPHA_INDICES]
N_ASPECTS = len(d3.ASPECTS)
N_EXTRA = N_ASPECTS * len(ALPHAS) * 2


def feature_names():
    names = []
    for aspect in d3.ASPECTS:
        for alpha in ALPHAS:
            tag = f"a{alpha:.7f}".rstrip("0").rstrip(".")
            names.extend((f"empema_{aspect}_{tag}",
                          f"absempema_{aspect}_{tag}"))
    return names


def _normal_score(reference, values):
    values = np.round(np.asarray(values, dtype=np.float64), 8)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0)
    )
    return ndtri(probability)


class EmpiricalEmaNull:
    def __init__(self, x_hist):
        hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        self.hs = hs
        self.scales = d15.EmaScales(
            x_hist, hs=hs, hist_inc=hist_inc
        )
        self.steady_scale = self.scales.scale(100000)[:, ALPHA_INDICES]
        self.reference = {}
        decay = 1.0 - ALPHAS
        for aspect_index, aspect in enumerate(d3.ASPECTS):
            g = np.asarray(hist_inc[aspect], dtype=np.float64) / hs.s[aspect]
            states = np.empty((g.size, len(ALPHAS)), dtype=np.float64)
            state = np.zeros(len(ALPHAS), dtype=np.float64)
            for index, value in enumerate(g):
                state = decay * state + ALPHAS * value
                states[index] = state
            for alpha_index, alpha in enumerate(ALPHAS):
                # Six effective time constants removes initialization bias,
                # while retaining at least 32 historical reference states.
                burn = max(32, int(np.ceil(6.0 / float(alpha))))
                burn = min(burn, max(g.size - 32, 0))
                reference = np.round(states[burn:, alpha_index], 8)
                self.reference[(aspect_index, alpha_index)] = np.sort(reference)
        ordered = [
            self.reference[(aspect_index, alpha_index)]
            for aspect_index in range(N_ASPECTS)
            for alpha_index in range(len(ALPHAS))
        ]
        self.reference_matrix, self.reference_lengths = (
            d22._pack_empirical_references(ordered)
        )

    def score(self, aspect_index, alpha_index, value):
        return _normal_score(
            self.reference[(aspect_index, alpha_index)], value
        )

    def probability(self, aspect_index, alpha_index, value):
        reference = self.reference[(aspect_index, alpha_index)]
        rounded = round(float(value), 8)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))

    def probabilities(self, values):
        probability = d22._rowwise_probability(
            self.reference_matrix,
            self.reference_lengths,
            values,
        )
        return probability.reshape(N_ASPECTS, len(ALPHAS))


def extra_features(x_hist, x_online):
    calibration = EmpiricalEmaNull(x_hist)
    hs, _, increments = d15._hist_and_online_increments(x_hist, x_online)
    steps = len(x_online)
    decay = 1.0 - ALPHAS
    state = np.zeros((N_ASPECTS, len(ALPHAS)), dtype=np.float64)
    states = np.empty((steps, N_ASPECTS, len(ALPHAS)), dtype=np.float64)
    matrix = np.column_stack([
        increments[aspect] / hs.s[aspect] for aspect in d3.ASPECTS
    ])
    for index in range(steps):
        state = decay[None, :] * state + ALPHAS[None, :] * matrix[index, :, None]
        states[index] = state
    finite_scale = calibration.scales.matrix(steps)[:, :, ALPHA_INDICES]
    adjusted = states / finite_scale * calibration.steady_scale[None, :, :]
    scores = np.empty_like(adjusted)
    for aspect_index in range(N_ASPECTS):
        for alpha_index in range(len(ALPHAS)):
            scores[:, aspect_index, alpha_index] = calibration.score(
                aspect_index, alpha_index,
                adjusted[:, aspect_index, alpha_index]
            )
    return np.stack((scores, np.abs(scores)), axis=-1).reshape(
        steps, N_EXTRA
    )


class StreamDetector26:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalEmaNull(x_hist)
        self.state = np.zeros((N_ASPECTS, len(ALPHAS)), dtype=np.float64)

    def _from_current_inner(self):
        n = self.inner.t
        position = (n - 1) % self.inner._wmax
        decay = 1.0 - ALPHAS
        for aspect_index, aspect in enumerate(d3.ASPECTS):
            value = (
                self.inner._ring[aspect][position] / self.inner.hs.s[aspect]
            )
            self.state[aspect_index] = (
                decay * self.state[aspect_index] + ALPHAS * value
            )
        finite_scale = self.calibration.scales.scale(n)[:, ALPHA_INDICES]
        adjusted = (
            self.state / finite_scale * self.calibration.steady_scale
        )
        probabilities = self.calibration.probabilities(adjusted)
        scores = ndtri(probabilities)
        return np.stack((scores, np.abs(scores)), axis=-1).ravel()

    def step(self, x):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(x)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(26)
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
            online[tau:] += rng.uniform(-1.0, 1.0)
            online[tau:] *= rng.uniform(1.1, 1.8)
        offline = extra_features(history, online)
        detector = StreamDetector26(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared = StreamDetector26(history, inner=shared_inner)
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
    detector = StreamDetector26(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_EXTRA
    print(
        f"OK: detectors26 stream == offline; {N_EXTRA} features; "
        f"{elapsed * 1e6:.0f} us/step"
    )

