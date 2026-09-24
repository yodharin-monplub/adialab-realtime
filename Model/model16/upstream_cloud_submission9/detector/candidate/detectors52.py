"""Empirical-null EWMA bank for higher-order ECDF shape modes.

Detectors3 already contains shifted-Legendre degrees 1--4.  Degrees 5--12
capture finer distributional shape, but earlier raw Fourier/suffix banks were
not cross-series stable.  This detector applies the accepted d26 mechanism:
four interleaved EWMA scales, full historical-autocovariance finite-start
correction, and a per-series empirical break-free-history null CDF.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3
import detectors15 as d15


DEGREES = tuple(range(5, 13))
ALPHA_INDICES = np.asarray((0, 2, 4, 6), dtype=np.int64)
ALPHAS = d15.ALPHAS[ALPHA_INDICES]
N_MODES = len(DEGREES)
N_FEATURES = N_MODES * len(ALPHAS) * 2


def feature_names():
    names = []
    for degree in DEGREES:
        for alpha in ALPHAS:
            tag = f"a{alpha:.7f}".rstrip("0").rstrip(".")
            names.extend((f"emphigh_L{degree}_{tag}",
                          f"absemphigh_L{degree}_{tag}"))
    return names


def _modes(u):
    """Ordinary Legendre P_5..P_12 at r=2F_hist(x)-1."""
    rank = 2.0 * np.asarray(u, dtype=np.float64) - 1.0
    previous = np.ones_like(rank)
    current = rank.copy()
    columns = []
    for degree in range(2, max(DEGREES) + 1):
        following = (((2 * degree - 1) * rank * current -
                      (degree - 1) * previous) / degree)
        previous, current = current, following
        if degree in DEGREES:
            columns.append(current.copy())
    return np.column_stack(columns)


class HighModeScales:
    def __init__(self, history_modes):
        values = np.asarray(history_modes, dtype=np.float64)
        self.var1 = np.empty(N_MODES, dtype=np.float64)
        self.ratio = np.empty((N_MODES, len(ALPHAS)), dtype=np.float64)
        for mode in range(N_MODES):
            variance, ratios = d15._steady_ratios(values[:, mode])
            self.var1[mode] = variance
            self.ratio[mode] = ratios[ALPHA_INDICES]

    def scale(self, n):
        n = int(n)
        decay = 1.0 - ALPHAS
        independent = (self.var1[:, None] * ALPHAS[None, :] /
                       (2.0 - ALPHAS[None, :]) *
                       (1.0 - decay[None, :] ** (2 * n)))
        ramp = 1.0 + (self.ratio - 1.0) * (
            1.0 - decay[None, :] ** max(n - 1, 0))
        return np.sqrt(np.maximum(independent * ramp, 1e-18))

    def matrix(self, length):
        n = np.arange(1, int(length) + 1, dtype=np.float64)
        decay = 1.0 - ALPHAS
        independent = (self.var1[None, :, None] *
                       ALPHAS[None, None, :] /
                       (2.0 - ALPHAS[None, None, :]) *
                       (1.0 - decay[None, None, :] **
                        (2 * n[:, None, None])))
        ramp = 1.0 + (self.ratio[None, :, :] - 1.0) * (
            1.0 - decay[None, None, :] **
            np.maximum(n[:, None, None] - 1, 0))
        return np.sqrt(np.maximum(independent * ramp, 1e-18))


def _normal_score(reference, values):
    values = np.round(np.asarray(values, dtype=np.float64), 8)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0))
    return ndtri(probability)


class EmpiricalHighModeNull:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.hs = d3.Hist3(history)
        raw = _modes(self.hs._cdf(history))
        self.center = raw.mean(axis=0)
        modes = raw - self.center
        self.scales = HighModeScales(modes)
        self.steady_scale = self.scales.scale(100000)
        decay = 1.0 - ALPHAS
        state = np.zeros((N_MODES, len(ALPHAS)), dtype=np.float64)
        states = np.empty((history.size, N_MODES, len(ALPHAS)),
                          dtype=np.float64)
        for index, row in enumerate(modes):
            state = decay[None, :] * state + ALPHAS[None, :] * row[:, None]
            states[index] = state
        self.reference = {}
        for mode in range(N_MODES):
            for alpha_index, alpha in enumerate(ALPHAS):
                burn = max(32, int(np.ceil(6.0 / float(alpha))))
                burn = min(burn, max(history.size - 32, 0))
                reference = np.round(states[burn:, mode, alpha_index], 8)
                self.reference[(mode, alpha_index)] = np.sort(reference)

    def probability(self, mode, alpha_index, value):
        reference = self.reference[(mode, alpha_index)]
        rounded = round(float(value), 8)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))


def extra_features(x_hist, x_online):
    calibration = EmpiricalHighModeNull(x_hist)
    online = np.asarray(x_online, dtype=np.float64)
    modes = _modes(calibration.hs._cdf(online)) - calibration.center
    decay = 1.0 - ALPHAS
    state = np.zeros((N_MODES, len(ALPHAS)), dtype=np.float64)
    states = np.empty((online.size, N_MODES, len(ALPHAS)),
                      dtype=np.float64)
    for index, row in enumerate(modes):
        state = decay[None, :] * state + ALPHAS[None, :] * row[:, None]
        states[index] = state
    adjusted = (states / calibration.scales.matrix(online.size) *
                calibration.steady_scale[None, :, :])
    scores = np.empty_like(adjusted)
    for mode in range(N_MODES):
        for alpha_index in range(len(ALPHAS)):
            scores[:, mode, alpha_index] = _normal_score(
                calibration.reference[(mode, alpha_index)],
                adjusted[:, mode, alpha_index])
    return np.stack((scores, np.abs(scores)), axis=-1).reshape(
        online.size, N_FEATURES)


class StreamDetector52:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalHighModeNull(x_hist)
        self.state = np.zeros((N_MODES, len(ALPHAS)), dtype=np.float64)

    def _from_current_inner(self):
        n = self.inner.t
        mode = (_modes(np.asarray([self.inner.u_prev]))[0] -
                self.calibration.center)
        decay = 1.0 - ALPHAS
        self.state = (decay[None, :] * self.state +
                      ALPHAS[None, :] * mode[:, None])
        adjusted = (self.state / self.calibration.scales.scale(n) *
                    self.calibration.steady_scale)
        probabilities = np.empty_like(adjusted)
        for mode_index in range(N_MODES):
            for alpha_index in range(len(ALPHAS)):
                probabilities[mode_index, alpha_index] = (
                    self.calibration.probability(
                        mode_index, alpha_index,
                        adjusted[mode_index, alpha_index]))
        scores = ndtri(probabilities)
        return np.stack((scores, np.abs(scores)), axis=-1).ravel()

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(52)
    worst = 0.0
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(10, 700))
        ar = float(rng.uniform(-0.85, 0.85))
        noise = rng.standard_t(float(rng.uniform(3.0, 12.0)),
                               n_hist + steps)
        values = np.empty(n_hist + steps, dtype=np.float64)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        history = values[:n_hist]
        online = values[n_hist:].copy()
        if trial % 2 == 0 and steps > 20:
            tau = int(rng.integers(5, steps - 5))
            post = online[tau:]
            online[tau:] = np.sign(post) * np.sqrt(np.abs(post))
        offline = extra_features(history, online)
        detector = StreamDetector52(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared = StreamDetector52(history, inner=shared_inner)
        shared_rows = []
        for value in online:
            shared_inner.step(value)
            shared_rows.append(shared.step_from_inner())
        shared_streamed = np.vstack(shared_rows)
        error = float(np.max(np.abs(offline - streamed)))
        shared_error = float(np.max(np.abs(offline - shared_streamed)))
        worst = max(worst, error, shared_error)
        assert offline.shape == (steps, N_FEATURES)
        assert error < 1e-9, (trial, error)
        assert shared_error < 1e-9, (trial, shared_error)
    assert len(feature_names()) == N_FEATURES
    detector = StreamDetector52(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    print(f"detectors52 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6:.0f} us/step")
