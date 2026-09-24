"""Empirical-null prequential regret for adaptive-vs-frozen predictors.

A permanent DGP change should eventually make a predictor that adapts online
outperform the model frozen on the break-free history.  This detector tracks
that causal excess predictive performance for mean, Gaussian variance, AR(1),
and AR(5) experts at several learning rates.  Per-series rolling history sums
map each regret statistic to a signed empirical normal score, making scores
comparable across heterogeneous series.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3


ALPHAS = np.asarray((0.125, 0.03125, 0.0078125), dtype=np.float64)
AR5_RATES = np.asarray((0.05, 0.01, 0.002), dtype=np.float64)
WINDOWS = (4, 8, 16, 32, 64, 128, 256, 512)
CHANNELS = (
    tuple(f"mean_a{alpha:g}" for alpha in ALPHAS)
    + tuple(f"variance_a{alpha:g}" for alpha in ALPHAS)
    + tuple(f"ar1_a{alpha:g}" for alpha in ALPHAS)
    + tuple(f"ar5_e{rate:g}" for rate in AR5_RATES)
)
N_CHANNELS = len(CHANNELS)
N_PER_CHANNEL = len(WINDOWS) + 2
N_EXTRA = N_CHANNELS * N_PER_CHANNEL


def feature_names():
    names = []
    for channel in CHANNELS:
        names.extend(f"regret_{channel}_w{window}" for window in WINDOWS)
        names.extend((f"regret_{channel}_maxabs",
                      f"regret_{channel}_signedmax"))
    return names


def _normal_score(reference, values):
    values = np.round(np.asarray(values, dtype=np.float64), 8)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0)
    )
    return ndtri(probability)


def _clip_regret(value):
    return min(25.0, max(-25.0, float(value)))


def _rescale_ar5(coefficients):
    norm = float(np.abs(coefficients).sum())
    if norm > 2.0:
        coefficients *= 2.0 / norm


def _regret_sequence(z, phi1, phi5, context):
    """Offline recurrence, independent of StreamDetector28."""
    values = np.asarray(z, dtype=np.float64)
    mean = np.zeros(len(ALPHAS), dtype=np.float64)
    variance = np.ones(len(ALPHAS), dtype=np.float64)
    adaptive_ar1 = np.full(len(ALPHAS), phi1, dtype=np.float64)
    adaptive_ar5 = np.tile(phi5, (len(AR5_RATES), 1)).astype(np.float64)
    lag = np.asarray(context[-d3.AR_P:], dtype=np.float64)[::-1].copy()
    output = np.empty((values.size, N_CHANNELS), dtype=np.float64)

    for timestep, current in enumerate(values):
        column = 0
        old_mean = mean.copy()
        errors = current - old_mean
        fixed_mean_loss = current * current
        for index in range(len(ALPHAS)):
            output[timestep, column] = _clip_regret(
                fixed_mean_loss - errors[index] * errors[index]
            )
            column += 1
        for index in range(len(ALPHAS)):
            v = float(np.clip(variance[index], 0.05, 20.0))
            adaptive_loss = np.log(v) + errors[index] * errors[index] / v
            output[timestep, column] = _clip_regret(
                fixed_mean_loss - adaptive_loss
            )
            column += 1

        previous = float(lag[0])
        fixed_ar1_error = current - phi1 * previous
        for index, alpha in enumerate(ALPHAS):
            adaptive_error = current - adaptive_ar1[index] * previous
            output[timestep, column] = _clip_regret(
                fixed_ar1_error * fixed_ar1_error
                - adaptive_error * adaptive_error
            )
            adaptive_ar1[index] += (
                alpha * adaptive_error * previous / (0.1 + previous * previous)
            )
            adaptive_ar1[index] = min(
                1.5, max(-1.5, float(adaptive_ar1[index]))
            )
            column += 1

        fixed_ar5_error = current - float(np.dot(phi5, lag))
        lag_norm = 0.1 + float(np.dot(lag, lag))
        for index, rate in enumerate(AR5_RATES):
            adaptive_error = current - float(np.dot(adaptive_ar5[index], lag))
            output[timestep, column] = _clip_regret(
                fixed_ar5_error * fixed_ar5_error
                - adaptive_error * adaptive_error
            )
            adaptive_ar5[index] += rate * adaptive_error * lag / lag_norm
            _rescale_ar5(adaptive_ar5[index])
            column += 1

        for index, alpha in enumerate(ALPHAS):
            mean[index] = old_mean[index] + alpha * errors[index]
            updated_variance = (
                (1.0 - alpha) * variance[index]
                + alpha * errors[index] * errors[index]
            )
            variance[index] = min(20.0, max(0.05, updated_variance))
        lag[1:] = lag[:-1]
        lag[0] = current
    return output


class RegretNull:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.mu = float(history.mean())
        sd = float(history.std())
        self.sd = sd if sd > 1e-12 else 1e-12
        z = (history - self.mu) / self.sd
        self.phi1 = float(d3._yule_walker(z, 1)[0])
        self.phi5 = d3._yule_walker(z, d3.AR_P)
        self.tail = (
            z[-d3.AR_P:].copy() if z.size >= d3.AR_P
            else np.concatenate([np.zeros(d3.AR_P - z.size), z])
        )
        # Circular lag context removes an arbitrary zero-padding transient
        # from the empirical null while using history only.
        regrets = _regret_sequence(z, self.phi1, self.phi5, self.tail)
        self.reference = {}
        for channel in range(N_CHANNELS):
            cumulative = np.concatenate(
                ([0.0], np.cumsum(regrets[:, channel]))
            )
            for window in WINDOWS:
                sums = cumulative[window:] - cumulative[:-window]
                self.reference[(channel, window)] = np.sort(
                    np.round(sums, 8)
                )

    def probability(self, channel, window, value):
        reference = self.reference[(channel, window)]
        rounded = round(float(value), 8)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))


def extra_features(x_hist, x_online):
    calibration = RegretNull(x_hist)
    values = (
        np.asarray(x_online, dtype=np.float64) - calibration.mu
    ) / calibration.sd
    regrets = _regret_sequence(
        values, calibration.phi1, calibration.phi5, calibration.tail
    )
    steps = values.size
    output = np.zeros((steps, N_EXTRA), dtype=np.float64)
    column = 0
    for channel in range(N_CHANNELS):
        cumulative = np.concatenate(
            ([0.0], np.cumsum(regrets[:, channel]))
        )
        scores = []
        for window in WINDOWS:
            score = np.zeros(steps, dtype=np.float64)
            if steps >= window:
                sums = cumulative[window:] - cumulative[:-window]
                score[window - 1:] = _normal_score(
                    calibration.reference[(channel, window)], sums
                )
            output[:, column] = score
            scores.append(score)
            column += 1
        matrix = np.column_stack(scores)
        index = np.argmax(np.abs(matrix), axis=1)
        output[:, column] = np.max(np.abs(matrix), axis=1)
        output[:, column + 1] = matrix[np.arange(steps), index]
        column += 2
    return output


class StreamDetector28:
    def __init__(self, x_hist):
        self.calibration = RegretNull(x_hist)
        self.mean = np.zeros(len(ALPHAS), dtype=np.float64)
        self.variance = np.ones(len(ALPHAS), dtype=np.float64)
        self.adaptive_ar1 = np.full(
            len(ALPHAS), self.calibration.phi1, dtype=np.float64
        )
        self.adaptive_ar5 = np.tile(
            self.calibration.phi5, (len(AR5_RATES), 1)
        ).astype(np.float64)
        self.lag = self.calibration.tail[::-1].copy()
        self.ring = np.zeros((N_CHANNELS, max(WINDOWS)), dtype=np.float64)
        self.sums = np.zeros((N_CHANNELS, len(WINDOWS)), dtype=np.float64)
        self.t = 0

    def _regrets(self, current):
        output = np.empty(N_CHANNELS, dtype=np.float64)
        column = 0
        old_mean = self.mean.copy()
        errors = current - old_mean
        fixed_mean_loss = current * current
        for index in range(len(ALPHAS)):
            output[column] = _clip_regret(
                fixed_mean_loss - errors[index] * errors[index]
            )
            column += 1
        for index in range(len(ALPHAS)):
            variance = float(np.clip(self.variance[index], 0.05, 20.0))
            adaptive_loss = (
                np.log(variance) + errors[index] * errors[index] / variance
            )
            output[column] = _clip_regret(fixed_mean_loss - adaptive_loss)
            column += 1

        previous = float(self.lag[0])
        fixed_error = current - self.calibration.phi1 * previous
        for index, alpha in enumerate(ALPHAS):
            adaptive_error = current - self.adaptive_ar1[index] * previous
            output[column] = _clip_regret(
                fixed_error * fixed_error - adaptive_error * adaptive_error
            )
            self.adaptive_ar1[index] += (
                alpha * adaptive_error * previous / (0.1 + previous * previous)
            )
            self.adaptive_ar1[index] = min(
                1.5, max(-1.5, float(self.adaptive_ar1[index]))
            )
            column += 1

        fixed_error = current - float(np.dot(self.calibration.phi5, self.lag))
        lag_norm = 0.1 + float(np.dot(self.lag, self.lag))
        for index, rate in enumerate(AR5_RATES):
            adaptive_error = current - float(
                np.dot(self.adaptive_ar5[index], self.lag)
            )
            output[column] = _clip_regret(
                fixed_error * fixed_error - adaptive_error * adaptive_error
            )
            self.adaptive_ar5[index] += (
                rate * adaptive_error * self.lag / lag_norm
            )
            _rescale_ar5(self.adaptive_ar5[index])
            column += 1

        for index, alpha in enumerate(ALPHAS):
            self.mean[index] = old_mean[index] + alpha * errors[index]
            updated_variance = (
                (1.0 - alpha) * self.variance[index]
                + alpha * errors[index] * errors[index]
            )
            self.variance[index] = min(
                20.0, max(0.05, updated_variance)
            )
        self.lag[1:] = self.lag[:-1]
        self.lag[0] = current
        return output

    def step(self, x):
        current = (float(x) - self.calibration.mu) / self.calibration.sd
        regrets = self._regrets(current)
        position = self.t % self.ring.shape[1]
        for window_index, window in enumerate(WINDOWS):
            self.sums[:, window_index] += regrets
            if self.t >= window:
                old_position = (self.t - window) % self.ring.shape[1]
                self.sums[:, window_index] -= self.ring[:, old_position]
        self.ring[:, position] = regrets
        self.t += 1

        probabilities = np.full(
            (N_CHANNELS, len(WINDOWS)), 0.5, dtype=np.float64
        )
        for channel in range(N_CHANNELS):
            for window_index, window in enumerate(WINDOWS):
                if self.t >= window:
                    probabilities[channel, window_index] = (
                        self.calibration.probability(
                        channel, window, self.sums[channel, window_index]
                    )
                    )
        score_matrix = ndtri(probabilities)
        output = np.empty(N_EXTRA, dtype=np.float64)
        for channel in range(N_CHANNELS):
            scores = score_matrix[channel]
            column = channel * N_PER_CHANNEL
            output[column:column + len(WINDOWS)] = scores
            index = int(np.argmax(np.abs(scores)))
            output[column + len(WINDOWS)] = float(np.max(np.abs(scores)))
            output[column + len(WINDOWS) + 1] = scores[index]
        return output


if __name__ == "__main__":
    rng = np.random.default_rng(28)
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(10, 700))
        ar = rng.uniform(-0.8, 0.8)
        noise = rng.standard_t(rng.uniform(3.0, 12.0), n_hist + steps)
        values = np.empty(n_hist + steps, dtype=np.float64)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        history, online = values[:n_hist], values[n_hist:].copy()
        if trial % 3 == 0 and steps > 30:
            tau = int(rng.integers(10, steps - 10))
            online[tau:] += rng.uniform(-1.2, 1.2)
        elif trial % 3 == 1 and steps > 30:
            tau = int(rng.integers(10, steps - 10))
            online[tau:] *= rng.uniform(1.2, 2.0)
        offline = extra_features(history, online)
        detector = StreamDetector28(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (steps, N_EXTRA)
        assert error < 1e-9, (trial, error)
    detector = StreamDetector28(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_EXTRA
    print(
        f"OK: detectors28 stream == offline; {N_EXTRA} features; "
        f"{elapsed * 1e6:.0f} us/step"
    )
