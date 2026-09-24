"""Empirical-null conditional AR-law generalized likelihood ratios.

This detector targets dependence changes while suppressing marginal effects.
Values are first mapped through the series' historical ECDF and Gaussianized.
For AR orders 1 and 5, and several suffix horizons, a null model with the
historical AR coefficients is compared with a locally refitted AR model.
Both models fit the same local intercept and residual variance, so local mean
and scale shifts largely cancel.  Every GLR is mapped through a per-series
break-free-history empirical null.

The offline path uses vectorized cumulative sufficient statistics.  The
streaming path independently maintains rolling sufficient statistics; the
module self-test requires their empirical scores to agree exactly.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3


ORDERS = (1, 5)
WINDOWS = (16, 32, 64, 128, 256)
MAX_REFERENCE = 127
RIDGE = 1e-6
N_RAW = len(ORDERS) * len(WINDOWS)
N_FEATURES = N_RAW + 4


def feature_names():
    names = [f"arcp_p{order}_w{window}"
             for order in ORDERS for window in WINDOWS]
    names.extend(("arcp_max", "arcp_top3", "arcp_top8", "arcp_l2"))
    return names


def _gaussianize(u, n_hist):
    epsilon = 0.5 / max(int(n_hist), 1)
    probability = np.clip(np.asarray(u, dtype=np.float64),
                          epsilon, 1.0 - epsilon)
    return ndtri(probability)


def _rows(values, order, context=None):
    values = np.asarray(values, dtype=np.float64)
    order = int(order)
    if context is None:
        full = values
        response = values[order:]
        predictors = np.column_stack([
            full[order - lag:full.size - lag]
            for lag in range(1, order + 1)
        ])
    else:
        tail = np.asarray(context, dtype=np.float64)[-order:]
        full = np.concatenate((tail, values))
        response = values
        predictors = np.column_stack([
            full[order - lag:order - lag + values.size]
            for lag in range(1, order + 1)
        ])
    return response, predictors


def _cumulative_sufficient(response, predictors):
    y = np.asarray(response, dtype=np.float64)
    x = np.asarray(predictors, dtype=np.float64)
    return {
        "y": np.concatenate(([0.0], np.cumsum(y))),
        "yy": np.concatenate(([0.0], np.cumsum(y * y))),
        "x": np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0))),
        "xy": np.vstack((np.zeros((1, x.shape[1])),
                          np.cumsum(x * y[:, None], axis=0))),
        "xx": np.concatenate((
            np.zeros((1, x.shape[1], x.shape[1])),
            np.cumsum(x[:, :, None] * x[:, None, :], axis=0)), axis=0),
    }


def _statistics_from_sufficient(n, sy, syy, sx, sxy, sxx, phi):
    """Batched or scalar conditional AR GLR from sufficient statistics."""
    n = np.asarray(n, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    syy = np.asarray(syy, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)
    sxy = np.asarray(sxy, dtype=np.float64)
    sxx = np.asarray(sxx, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    centered_xx = sxx - sx[..., :, None] * sx[..., None, :] / n[..., None, None]
    centered_xy = sxy - sx * sy[..., None] / n[..., None]
    centered_yy = syy - sy * sy / n

    null_sum = sy - np.einsum("...i,i->...", sx, phi)
    null_square = (syy - 2.0 * np.einsum("...i,i->...", sxy, phi) +
                   np.einsum("i,...ij,j->...", phi, sxx, phi))
    null_rss = null_square - null_sum * null_sum / n

    dimension = phi.size
    trace = np.trace(centered_xx, axis1=-2, axis2=-1)
    penalty = RIDGE * np.maximum(trace / max(dimension, 1), 1.0)
    regularized = centered_xx + penalty[..., None, None] * np.eye(dimension)
    try:
        coefficients = np.linalg.solve(regularized, centered_xy[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coefficients = np.zeros_like(centered_xy)
    explained = np.einsum("...i,...i->...", centered_xy, coefficients)
    alternative_rss = centered_yy - explained
    epsilon = np.maximum(1e-10 * n, 1e-12)
    null_rss = np.maximum(null_rss, epsilon)
    alternative_rss = np.maximum(alternative_rss, epsilon)
    statistic = n * np.log(null_rss / alternative_rss)
    return np.maximum(statistic, 0.0)


def _window_statistics(response, predictors, phi, windows, endpoints=None):
    cumulative = _cumulative_sufficient(response, predictors)
    total = len(response)
    output = {}
    for window in windows:
        if total < window:
            output[window] = (np.empty(0, dtype=np.int64),
                              np.empty(0, dtype=np.float64))
            continue
        if endpoints is None:
            end = np.arange(window, total + 1, dtype=np.int64)
        else:
            end = np.asarray(endpoints, dtype=np.int64)
            end = end[(end >= window) & (end <= total)]
        start = end - int(window)
        sy = cumulative["y"][end] - cumulative["y"][start]
        syy = cumulative["yy"][end] - cumulative["yy"][start]
        sx = cumulative["x"][end] - cumulative["x"][start]
        sxy = cumulative["xy"][end] - cumulative["xy"][start]
        sxx = cumulative["xx"][end] - cumulative["xx"][start]
        values = _statistics_from_sufficient(
            np.full(end.size, float(window)), sy, syy, sx, sxy, sxx, phi)
        output[window] = end, values
    return output


def _normal_score(reference, value):
    value = np.round(np.asarray(value, dtype=np.float64), 8)
    left = np.searchsorted(reference, value, side="left")
    right = np.searchsorted(reference, value, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0))
    return ndtri(probability)


class EmpiricalARChangeNull:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.hs = d3.Hist3(history)
        gaussian = _gaussianize(self.hs._cdf(history), self.hs.n_hist)
        self.tail = gaussian[-max(ORDERS):].copy()
        self.phi = {}
        self.reference = {}
        for order in ORDERS:
            phi = d3._yule_walker(gaussian, order)
            self.phi[order] = phi
            response, predictors = _rows(gaussian, order)
            total = response.size
            for window in WINDOWS:
                possible = total - window + 1
                if possible <= 0:
                    reference = np.asarray([0.0])
                else:
                    count = min(MAX_REFERENCE, possible)
                    endpoints = np.unique(np.linspace(
                        window, total, count, dtype=np.int64))
                    _, values = _window_statistics(
                        response, predictors, phi, (window,), endpoints)[window]
                    reference = np.sort(np.round(values, 8))
                self.reference[(order, window)] = reference

    def score(self, order, window, value):
        return _normal_score(self.reference[(order, window)], value)

    def probability(self, order, window, value):
        reference = self.reference[(order, window)]
        rounded = round(float(value), 8)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))


def _summaries(raw):
    raw = np.asarray(raw, dtype=np.float64)
    ordered = np.sort(raw, axis=1)
    return np.column_stack((
        ordered[:, -1], ordered[:, -3:].mean(axis=1),
        ordered[:, -8:].mean(axis=1),
        np.sqrt(np.mean(np.maximum(raw, 0.0) ** 2, axis=1)),
    ))


def extra_features(x_hist, x_online):
    calibration = EmpiricalARChangeNull(x_hist)
    online = np.asarray(x_online, dtype=np.float64)
    gaussian = _gaussianize(
        calibration.hs._cdf(online), calibration.hs.n_hist)
    output = np.zeros((online.size, N_RAW), dtype=np.float64)
    column = 0
    for order in ORDERS:
        response, predictors = _rows(
            gaussian, order, calibration.tail)
        results = _window_statistics(
            response, predictors, calibration.phi[order], WINDOWS)
        for window in WINDOWS:
            endpoints, values = results[window]
            if endpoints.size:
                output[endpoints - 1, column] = calibration.score(
                    order, window, values)
            column += 1
    return np.column_stack((output, _summaries(output)))


class _RollingSufficient:
    def __init__(self, order, window, phi):
        self.order = int(order)
        self.window = int(window)
        self.phi = np.asarray(phi, dtype=np.float64)
        self.y = np.zeros(self.window, dtype=np.float64)
        self.x = np.zeros((self.window, self.order), dtype=np.float64)
        self.count = 0
        self.sy = 0.0
        self.syy = 0.0
        self.sx = np.zeros(self.order, dtype=np.float64)
        self.sxy = np.zeros(self.order, dtype=np.float64)
        self.sxx = np.zeros((self.order, self.order), dtype=np.float64)

    def step(self, response, predictors):
        position = self.count % self.window
        if self.count >= self.window:
            old_y = self.y[position]
            old_x = self.x[position]
            self.sy -= old_y
            self.syy -= old_y * old_y
            self.sx -= old_x
            self.sxy -= old_x * old_y
            self.sxx -= np.outer(old_x, old_x)
        predictors = np.asarray(predictors, dtype=np.float64)
        response = float(response)
        self.y[position] = response
        self.x[position] = predictors
        self.sy += response
        self.syy += response * response
        self.sx += predictors
        self.sxy += predictors * response
        self.sxx += np.outer(predictors, predictors)
        self.count += 1
        if self.count < self.window:
            return None
        return float(_statistics_from_sufficient(
            float(self.window), self.sy, self.syy, self.sx,
            self.sxy, self.sxx, self.phi))


class StreamDetector53:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalARChangeNull(x_hist)
        self.lags = list(self.calibration.tail[::-1])
        self.states = {
            (order, window): _RollingSufficient(
                order, window, self.calibration.phi[order])
            for order in ORDERS for window in WINDOWS
        }

    def _from_current_inner(self):
        probability = self.inner.u_prev
        current = float(_gaussianize(
            np.asarray([probability]), self.calibration.hs.n_hist)[0])
        raw = np.zeros(N_RAW, dtype=np.float64)
        column = 0
        for order in ORDERS:
            predictors = np.asarray(self.lags[:order], dtype=np.float64)
            for window in WINDOWS:
                statistic = self.states[(order, window)].step(
                    current, predictors)
                if statistic is not None:
                    raw[column] = ndtri(self.calibration.probability(
                        order, window, statistic))
                column += 1
        self.lags = [current] + self.lags[:-1]
        ordered = np.sort(raw)
        return np.concatenate((raw, np.asarray((
            ordered[-1], ordered[-3:].mean(), ordered[-8:].mean(),
            np.sqrt(np.mean(np.maximum(raw, 0.0) ** 2)),
        ))))

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(53)
    worst = 0.0
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(20, 700))
        total = n_hist + steps
        values = np.zeros(total, dtype=np.float64)
        noise = rng.standard_t(float(rng.uniform(4.0, 12.0)), total)
        before = float(rng.uniform(-0.75, 0.75))
        after = float(np.clip(before + rng.uniform(-0.7, 0.7), -0.9, 0.9))
        change = n_hist + steps // 2
        for index in range(1, total):
            coefficient = (after if trial % 2 == 0 and index >= change
                           else before)
            values[index] = coefficient * values[index - 1] + noise[index]
        history = values[:n_hist]
        online = values[n_hist:]
        offline = extra_features(history, online)
        detector = StreamDetector53(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared = StreamDetector53(history, inner=shared_inner)
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
    detector = StreamDetector53(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    print(f"detectors53 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6:.0f} us/step")
