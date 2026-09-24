"""Empirical-null Shiryaev--Roberts evidence for mean/scale changes.

The accepted banks use hard maxima (CUSUM) and fixed/EWMA suffix geometry.
Here each recursion softly sums likelihood evidence over *every* possible
change time.  Separate Gaussian mean and scale alternatives run on the raw
location innovation and the AR(5)-whitened residual innovation.  Each state
is mapped through its own empirical break-free-history null, preserving
cross-series comparability without assuming Gaussian histories are exact.

Both offline and streaming paths initialize from the historical-null median,
which removes arbitrary finite-start differences while injecting no label or
online information.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3
import detectors15 as d15


CHANNELS = ("z", "e")
MEAN_MAGNITUDES = (0.125, 0.25, 0.5, 1.0)
MEAN_SHIFTS = np.asarray(
    [value for magnitude in MEAN_MAGNITUDES
     for value in (magnitude, -magnitude)], dtype=np.float64)
SCALE_RATIOS = np.asarray((0.50, 0.75, 1.25, 1.50, 2.00),
                          dtype=np.float64)
N_ALTERNATIVES = len(MEAN_SHIFTS) + len(SCALE_RATIOS)
N_QUANTILES = 127
QUANTILE_PROBABILITIES = (
    np.arange(1, N_QUANTILES + 1, dtype=np.float64) /
    float(N_QUANTILES + 1))
N_RAW = len(CHANNELS) * N_ALTERNATIVES
N_SUMMARY = 4
N_FEATURES = N_RAW + N_SUMMARY


def feature_names():
    names = []
    for channel in CHANNELS:
        for shift in MEAN_SHIFTS:
            names.append(f"sr_{channel}_mean_{shift:+.3f}")
        for ratio in SCALE_RATIOS:
            names.append(f"sr_{channel}_scale_{ratio:.2f}")
    names.extend(("sr_max", "sr_top3", "sr_top8", "sr_pos_l2"))
    return names


def _log_likelihood_ratios(values):
    """Return Gaussian alternative/null log likelihood ratios."""
    values = np.asarray(values, dtype=np.float64)
    mean = (values[..., None] * MEAN_SHIFTS -
            0.5 * MEAN_SHIFTS * MEAN_SHIFTS)
    ratio = SCALE_RATIOS
    scale = (-np.log(ratio) + 0.5 * values[..., None] ** 2 *
             (1.0 - 1.0 / (ratio * ratio)))
    return np.concatenate((mean, scale), axis=-1)


def _state_path(matrix, initial=None):
    """Shiryaev--Roberts log state for a (time, channel) matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    state = (np.zeros((len(CHANNELS), N_ALTERNATIVES), dtype=np.float64)
             if initial is None else np.asarray(initial,
                                                dtype=np.float64).copy())
    output = np.empty(
        (matrix.shape[0], len(CHANNELS), N_ALTERNATIVES),
        dtype=np.float64)
    likelihood = _log_likelihood_ratios(matrix)
    for index in range(matrix.shape[0]):
        # log R_t = log(1 + R_{t-1}) + log LR_t.  Clipping only prevents
        # floating overflow on blatant breaks; the empirical-null range is
        # far below the ceiling.
        state = np.logaddexp(0.0, state) + likelihood[index]
        state = np.clip(state, -80.0, 200.0)
        output[index] = state
    return output


def _normalized_channels(hs, increments):
    columns = [
        np.asarray(increments[channel], dtype=np.float64) / hs.s[channel]
        for channel in CHANNELS
    ]
    # Historical AR residuals deliberately discard AR_P warm-up values.
    # Align every calibration channel to that common causal tail.  Online
    # arrays already share one length, so this is a no-op there.
    length = min(column.size for column in columns)
    return np.column_stack([column[-length:] for column in columns])


def _normal_score(reference, values):
    values = np.round(np.asarray(values, dtype=np.float64), 8)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0))
    return ndtri(probability)


class EmpiricalSRNull:
    def __init__(self, x_hist):
        hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        self.hs = hs
        history = _normalized_channels(hs, hist_inc)
        states = _state_path(history)
        # The weakest mean alternative has a null memory on the order of
        # 1/delta^2=64; eight such scales removes initialization bias.
        burn = min(512, max(history.shape[0] - 128, 0))
        stable = states[burn:]
        self.initial = np.median(stable, axis=0)
        self.reference = np.round(np.quantile(
            stable, QUANTILE_PROBABILITIES, axis=0), 8).transpose(1, 2, 0)

    def score_matrix(self, states):
        states = np.asarray(states, dtype=np.float64)
        output = np.empty_like(states)
        for channel in range(len(CHANNELS)):
            for alternative in range(N_ALTERNATIVES):
                output[:, channel, alternative] = _normal_score(
                    self.reference[channel, alternative],
                    states[:, channel, alternative])
        return output

    def probabilities(self, state):
        state = np.round(np.asarray(state, dtype=np.float64), 8)
        # Fixed 127-quantile references make this vectorized comparison cheap
        # enough for every streaming step.
        left = np.sum(
            state[:, :, None] > self.reference, axis=2)
        right = np.sum(
            state[:, :, None] >= self.reference, axis=2)
        return (left + right + 1.0) / float(2 * (N_QUANTILES + 1))


def _with_summaries(score_matrix):
    flat = np.asarray(score_matrix, dtype=np.float64).reshape(
        score_matrix.shape[0], N_RAW)
    ordered = np.sort(flat, axis=1)
    positive = np.maximum(flat, 0.0)
    summaries = np.column_stack((
        ordered[:, -1],
        ordered[:, -3:].mean(axis=1),
        ordered[:, -8:].mean(axis=1),
        np.sqrt(np.mean(positive * positive, axis=1)),
    ))
    return np.column_stack((flat, summaries))


def extra_features(x_hist, x_online):
    calibration = EmpiricalSRNull(x_hist)
    hs, _, increments = d15._hist_and_online_increments(x_hist, x_online)
    matrix = _normalized_channels(hs, increments)
    states = _state_path(matrix, calibration.initial)
    return _with_summaries(calibration.score_matrix(states))


class StreamDetector51:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalSRNull(x_hist)
        self.state = self.calibration.initial.copy()

    def _from_current_inner(self):
        position = (self.inner.t - 1) % self.inner._wmax
        values = np.asarray([
            self.inner._ring[channel][position] /
            self.inner.hs.s[channel]
            for channel in CHANNELS
        ], dtype=np.float64)
        likelihood = _log_likelihood_ratios(values)
        self.state = np.logaddexp(0.0, self.state) + likelihood
        self.state = np.clip(self.state, -80.0, 200.0)
        scores = ndtri(self.calibration.probabilities(self.state)).ravel()
        ordered = np.sort(scores)
        positive = np.maximum(scores, 0.0)
        return np.concatenate((scores, np.asarray((
            ordered[-1], ordered[-3:].mean(), ordered[-8:].mean(),
            np.sqrt(np.mean(positive * positive)),
        ))))

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(51)
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
            online[tau:] += float(rng.uniform(-1.0, 1.0))
            online[tau:] *= float(rng.uniform(0.7, 1.6))
        offline = extra_features(history, online)
        detector = StreamDetector51(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared = StreamDetector51(history, inner=shared_inner)
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
    detector = StreamDetector51(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    print(f"detectors51 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6:.0f} us/step")
