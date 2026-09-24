"""Random-feature nonlinear forecast-surprise detector.

This is an independent companion to detectors57.  It preserves the same
strict fit/calibration split and evidence geometry, but replaces the fixed
polynomial forecaster with deterministic random Fourier features of the six
lagged Gaussian-rank values.  The random seed, frequencies, ridge penalty,
and feature count are constants, so the detector is fully deterministic.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3
import detectors57 as d57


ORDER = d57.ORDER
WINDOWS = d57.WINDOWS
ASPECTS = d57.ASPECTS
BLOCK = d57.BLOCK
N_EXTRA = d57.N_EXTRA
RFF_PER_SCALE = 12
RFF_SCALES = (0.5, 1.0, 2.0)
RIDGE = 50.0

_RNG = np.random.default_rng(5801)
_FREQUENCY = np.vstack([
    scale * _RNG.normal(size=(RFF_PER_SCALE, ORDER))
    for scale in RFF_SCALES
])
_PHASE = _RNG.uniform(0.0, 2.0 * np.pi, _FREQUENCY.shape[0])


def feature_names():
    return [name.replace("nl_", "rff_", 1)
            for name in d57.feature_names()]


def _basis(lags):
    lags = np.asarray(lags, dtype=np.float64)
    clipped = np.clip(lags, -4.0, 4.0)
    projection = clipped @ _FREQUENCY.T + _PHASE
    random = np.sqrt(2.0) * np.cos(projection)
    return np.column_stack((clipped, random))


def _fit_predictor(train_lags, train_target):
    raw = _basis(train_lags)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    design = (raw - mean) / scale
    target_mean = float(np.mean(train_target))
    centered = np.asarray(train_target, dtype=np.float64) - target_mean
    gram = design.T @ design
    rhs = design.T @ centered
    try:
        coefficient = np.linalg.solve(
            gram + RIDGE * np.eye(design.shape[1]), rhs)
    except np.linalg.LinAlgError:
        coefficient = np.zeros(design.shape[1], dtype=np.float64)
    if not np.all(np.isfinite(coefficient)):
        coefficient[:] = 0.0
    return mean, scale, target_mean, coefficient


def _predict(lags, mean, scale, target_mean, coefficient):
    return target_mean + ((_basis(lags) - mean) / scale) @ coefficient


class RandomFeatureNull:
    increments = d57.NonlinearNull.increments

    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.hs = d3.Hist3(history)
        gaussian = d57._gaussianize(
            self.hs._cdf(history), self.hs.n_hist)
        target, lags = d57._lag_rows(gaussian)
        n_rows = target.size
        split = int(np.floor(d57.TRAIN_FRACTION * n_rows))
        split = min(max(split, 128), max(n_rows - 128, 128))
        if n_rows < 256:
            split = max(1, n_rows // 2)
        self.mean, self.scale, self.target_mean, self.coefficient = \
            _fit_predictor(lags[:split], target[:split])
        prediction = _predict(
            lags[split:], self.mean, self.scale,
            self.target_mean, self.coefficient)
        residual = target[split:] - prediction
        if residual.size < 16:
            residual = target - _predict(
                lags, self.mean, self.scale,
                self.target_mean, self.coefficient)
        self.residual_mean = float(np.mean(residual))
        residual_sd = float(np.std(residual))
        self.residual_sd = residual_sd if residual_sd > 1e-8 else 1.0
        standardized = (residual - self.residual_mean) / self.residual_sd
        self.sorted_residual = np.sort(residual)
        self.gaussian_tail = gaussian[-ORDER:].copy()
        self.residual_tail = (standardized[-2:].copy()
                              if standardized.size >= 2
                              else np.zeros(2, dtype=np.float64))

        probability = d57._ecdf(self.sorted_residual, residual)
        legendre = d57._legendre(probability)
        self.center = {
            "nl_e": float(np.mean(standardized)),
            "nl_e2": float(np.mean(standardized * standardized)),
            "nl_abse": float(np.mean(np.abs(standardized))),
            "nl_L1": float(np.mean(legendre[0])),
            "nl_L2": float(np.mean(legendre[1])),
            "nl_L3": float(np.mean(legendre[2])),
            "nl_L4": float(np.mean(legendre[3])),
            "nl_tail_up": float(np.mean(probability > 0.95)),
            "nl_tail_dn": float(np.mean(probability < 0.05)),
            "nl_ac1": (float(np.mean(standardized[1:] * standardized[:-1]))
                       if standardized.size > 1 else 0.0),
            "nl_ac2": (float(np.mean(standardized[2:] * standardized[:-2]))
                       if standardized.size > 2 else 0.0),
        }
        increments = self.increments(
            standardized, probability, history_mode=True)
        self.null_sd = {}
        for aspect in ASPECTS:
            value = float(np.std(increments[aspect]))
            self.null_sd[aspect] = value if value > 1e-8 else 1.0

    def predict_one(self, lag_values):
        return float(_predict(
            np.asarray(lag_values, dtype=np.float64)[None, :],
            self.mean, self.scale, self.target_mean,
            self.coefficient)[0])

    def residual_probability(self, residual):
        left = self.sorted_residual.searchsorted(residual, side="left")
        right = self.sorted_residual.searchsorted(residual, side="right")
        return float((left + right) / (2.0 * self.sorted_residual.size))


def extra_features(x_hist, x_online):
    null = RandomFeatureNull(x_hist)
    online = np.asarray(x_online, dtype=np.float64)
    probability = null.hs._cdf(online)
    gaussian = d57._gaussianize(probability, null.hs.n_hist)
    full = np.concatenate((null.gaussian_tail, gaussian))
    lags = np.column_stack([
        full[ORDER - lag:ORDER - lag + online.size]
        for lag in range(1, ORDER + 1)
    ])
    residual = gaussian - _predict(
        lags, null.mean, null.scale,
        null.target_mean, null.coefficient)
    standardized = (residual - null.residual_mean) / null.residual_sd
    residual_probability = d57._ecdf(null.sorted_residual, residual)
    increments = null.increments(
        standardized, residual_probability,
        previous=null.residual_tail)

    steps = online.size
    count = np.arange(1, steps + 1, dtype=np.float64)
    index = np.arange(1, steps + 1)
    columns = []
    for aspect in ASPECTS:
        value = increments[aspect]
        sd = null.null_sd[aspect]
        cumulative = np.cumsum(value)
        padded = np.concatenate(([0.0], cumulative))
        running_min = np.minimum.accumulate(np.minimum(cumulative, 0.0))
        running_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        columns.extend((
            cumulative / (sd * np.sqrt(count)),
            (cumulative - running_min) / (sd * np.sqrt(count)),
            (running_max - cumulative) / (sd * np.sqrt(count)),
        ))
        for window in WINDOWS:
            start = np.maximum(0, index - window)
            effective = np.minimum(count, window)
            total = padded[1:] - padded[start]
            columns.append(total / (sd * np.sqrt(effective)))
    return np.column_stack(columns)


class StreamDetector58(d57.StreamDetector57):
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.null = RandomFeatureNull(x_hist)
        self.gaussian_lags = list(self.null.gaussian_tail[::-1])
        self.residual_lags = list(self.null.residual_tail[::-1])
        self.t = 0
        self.cumulative = {aspect: 0.0 for aspect in ASPECTS}
        self.minimum = {aspect: 0.0 for aspect in ASPECTS}
        self.maximum = {aspect: 0.0 for aspect in ASPECTS}
        self.ring = {aspect: np.zeros(max(WINDOWS), dtype=np.float64)
                     for aspect in ASPECTS}
        self.window_sum = {
            aspect: {window: 0.0 for window in WINDOWS}
            for aspect in ASPECTS
        }


if __name__ == "__main__":
    rng = np.random.default_rng(58)
    worst = 0.0
    for trial in range(5):
        n_hist = int(rng.integers(1000, 2600))
        steps = int(rng.integers(20, 600))
        total = n_hist + steps
        values = np.zeros(total, dtype=np.float64)
        noise = rng.standard_t(float(rng.uniform(4.0, 12.0)), total)
        before = float(rng.uniform(-0.6, 0.6))
        after = float(np.clip(before + rng.uniform(-0.8, 0.8), -0.9, 0.9))
        change = n_hist + steps // 2
        for index in range(2, total):
            coefficient = after if index >= change else before
            nonlinear = 0.18 * np.sin(values[index - 2])
            values[index] = (coefficient * values[index - 1]
                             + nonlinear + noise[index])
        history = values[:n_hist]
        online = values[n_hist:]
        offline = extra_features(history, online)
        detector = StreamDetector58(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared_detector = StreamDetector58(history, inner=shared_inner)
        shared_rows = []
        for value in online:
            shared_inner.step(value)
            shared_rows.append(shared_detector.step_from_inner())
        shared = np.vstack(shared_rows)
        error = float(np.max(np.abs(offline - streamed)))
        shared_error = float(np.max(np.abs(offline - shared)))
        worst = max(worst, error, shared_error)
        assert offline.shape == (steps, N_EXTRA)
        assert error < 1e-9, (trial, error)
        assert shared_error < 1e-9, (trial, shared_error)
    assert len(feature_names()) == N_EXTRA
    detector = StreamDetector58(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000.0
    print(f"detectors58 OK: {N_EXTRA} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6:.0f} us/step")
