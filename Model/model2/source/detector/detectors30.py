"""Empirical-null EWMA evidence for the seven accepted detectors4 aspects.

The packaged d4x specialist uses fixed suffix windows on extreme 1% tail and
multi-lag difference-energy increments.  This module applies the complementary
EWMA geometry proven by detectors15, then maps each filtered state through its
own per-series break-free-history empirical null as in detectors26.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors4 as d4
import detectors15 as d15


ASPECTS = ("x99_up", "x99_dn", "d1e", "d2e", "d4e", "d8e", "absd1")
ALPHA_INDICES = np.asarray((0, 2, 4, 6), dtype=np.int64)
ALPHAS = d15.ALPHAS[ALPHA_INDICES]
N_ASPECTS = len(ASPECTS)
N_EXTRA = N_ASPECTS * len(ALPHAS) * 2
MAX_ACOV_LAG = 512
MIN_RATIO = 0.10
MAX_RATIO = 10.0


def feature_names():
    names = []
    for aspect in ASPECTS:
        for alpha in ALPHAS:
            tag = f"a{alpha:.7f}".rstrip("0").rstrip(".")
            names.extend((f"d4empema_{aspect}_{tag}",
                          f"d4absempema_{aspect}_{tag}"))
    return names


def _hist_components(x_hist):
    history = np.asarray(x_hist, dtype=np.float64)
    hs = d4.Hist4(history)
    z = (history - hs.mu0) / hs.sd0
    u = hs._cdf(history)
    e = d4._ar_residuals(z, hs.phi, np.zeros(d4.AR_P))
    e = e[d4.AR_P:] if e.size > d4.AR_P else e
    if e.size < 10:
        e = z
    increments = d4._aspect_increments_hist(hs, z, u, e)
    return hs, increments


def _steady_ratio(values):
    x = np.asarray(values, dtype=np.float64)
    x = x - float(np.mean(x))
    gamma0 = float(np.mean(x * x))
    if x.size < 16 or not np.isfinite(gamma0) or gamma0 < 1e-18:
        return gamma0 if gamma0 > 0 else 1.0, np.ones(len(ALPHAS))
    max_lag = min(MAX_ACOV_LAG, x.size - 1)
    nfft = 1 << (2 * x.size - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    raw = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=nfft
    )[:max_lag + 1]
    gamma = raw / np.arange(x.size, x.size - max_lag - 1, -1)
    lag = np.arange(1, max_lag + 1, dtype=np.float64)
    ratios = np.empty(len(ALPHAS), dtype=np.float64)
    for index, alpha in enumerate(ALPHAS):
        ratio = 1.0 + 2.0 * np.sum(
            ((1.0 - alpha) ** lag) * gamma[1:]
        ) / gamma0
        if not np.isfinite(ratio):
            ratio = 1.0
        ratios[index] = np.clip(ratio, MIN_RATIO, MAX_RATIO)
    return gamma0, ratios


class EmpiricalD4EmaNull:
    def __init__(self, x_hist):
        self.hs, hist_inc = _hist_components(x_hist)
        self.var1 = np.empty(N_ASPECTS, dtype=np.float64)
        self.ratio = np.empty((N_ASPECTS, len(ALPHAS)), dtype=np.float64)
        self.reference = {}
        decay = 1.0 - ALPHAS
        for aspect_index, aspect in enumerate(ASPECTS):
            increments = (
                np.asarray(hist_inc[aspect], dtype=np.float64) /
                self.hs.s[aspect]
            )
            self.var1[aspect_index], self.ratio[aspect_index] = (
                _steady_ratio(increments)
            )
            states = np.empty((increments.size, len(ALPHAS)), dtype=np.float64)
            state = np.zeros(len(ALPHAS), dtype=np.float64)
            for index, value in enumerate(increments):
                state = decay * state + ALPHAS * value
                states[index] = state
            for alpha_index, alpha in enumerate(ALPHAS):
                burn = max(32, int(np.ceil(6.0 / float(alpha))))
                burn = min(burn, max(increments.size - 32, 0))
                self.reference[(aspect_index, alpha_index)] = np.sort(
                    np.round(states[burn:, alpha_index], 8)
                )
        self.steady_scale = self.scale(100000)

    def scale(self, n):
        n = int(n)
        decay = 1.0 - ALPHAS
        independent = (
            self.var1[:, None] * ALPHAS[None, :] /
            (2.0 - ALPHAS[None, :]) *
            (1.0 - decay[None, :] ** (2 * n))
        )
        ramp = 1.0 + (self.ratio - 1.0) * (
            1.0 - decay[None, :] ** max(n - 1, 0)
        )
        return np.sqrt(np.maximum(independent * ramp, 1e-18))

    def scale_matrix(self, length):
        n = np.arange(1, int(length) + 1, dtype=np.float64)
        decay = 1.0 - ALPHAS
        independent = (
            self.var1[None, :, None] * ALPHAS[None, None, :] /
            (2.0 - ALPHAS[None, None, :]) *
            (1.0 - decay[None, None, :] ** (2 * n[:, None, None]))
        )
        ramp = 1.0 + (self.ratio[None, :, :] - 1.0) * (
            1.0 - decay[None, None, :] **
            np.maximum(n[:, None, None] - 1, 0)
        )
        return np.sqrt(np.maximum(independent * ramp, 1e-18))

    def score(self, aspect_index, alpha_index, values):
        reference = self.reference[(aspect_index, alpha_index)]
        rounded = np.round(np.asarray(values, dtype=np.float64), 8)
        left = np.searchsorted(reference, rounded, side="left")
        right = np.searchsorted(reference, rounded, side="right")
        probability = (left + right + 1.0) / (
            2.0 * (reference.size + 1.0)
        )
        return ndtri(probability)

    def probability(self, aspect_index, alpha_index, value):
        reference = self.reference[(aspect_index, alpha_index)]
        rounded = round(float(value), 8)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))


def extra_features(x_hist, x_online):
    calibration = EmpiricalD4EmaNull(x_hist)
    values = np.asarray(x_online, dtype=np.float64)
    z = (values - calibration.hs.mu0) / calibration.hs.sd0
    u = calibration.hs._cdf(values)
    e = d4._ar_residuals(
        z, calibration.hs.phi, calibration.hs.zh_tail[-d4.AR_P:]
    )
    increments = d4._aspect_increments_online(calibration.hs, z, u, e)
    matrix = np.column_stack([
        increments[aspect] / calibration.hs.s[aspect]
        for aspect in ASPECTS
    ])
    steps = values.size
    decay = 1.0 - ALPHAS
    state = np.zeros((N_ASPECTS, len(ALPHAS)), dtype=np.float64)
    states = np.empty((steps, N_ASPECTS, len(ALPHAS)), dtype=np.float64)
    for index in range(steps):
        state = decay[None, :] * state + ALPHAS[None, :] * matrix[index, :, None]
        states[index] = state
    adjusted = (
        states / calibration.scale_matrix(steps) *
        calibration.steady_scale[None, :, :]
    )
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


class StreamDetector30:
    def __init__(self, x_hist, inner=None):
        self.inner = d4.StreamDetector4(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalD4EmaNull(x_hist)
        self.state = np.zeros((N_ASPECTS, len(ALPHAS)), dtype=np.float64)

    def _from_current_inner(self):
        n = self.inner.t
        position = (n - 1) % self.inner._wmax
        decay = 1.0 - ALPHAS
        for aspect_index, aspect in enumerate(ASPECTS):
            value = (
                self.inner._ring[aspect][position] /
                self.inner.hs.s[aspect]
            )
            self.state[aspect_index] = (
                decay * self.state[aspect_index] + ALPHAS * value
            )
        adjusted = (
            self.state / self.calibration.scale(n) *
            self.calibration.steady_scale
        )
        probabilities = np.empty_like(adjusted)
        for aspect_index in range(N_ASPECTS):
            for alpha_index in range(len(ALPHAS)):
                probabilities[aspect_index, alpha_index] = (
                    self.calibration.probability(
                        aspect_index, alpha_index,
                        adjusted[aspect_index, alpha_index]
                    )
                )
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
    rng = np.random.default_rng(30)
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
        detector = StreamDetector30(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d4.StreamDetector4(history)
        shared = StreamDetector30(history, inner=shared_inner)
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
    detector = StreamDetector30(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_EXTRA
    print(
        f"OK: detectors30 stream == offline; {N_EXTRA} features; "
        f"{elapsed * 1e6:.0f} us/step"
    )
