"""Causal nonlinear forecast-surprise detector.

The accepted detector bank models linear dependence with an AR(5).  This
module targets a distinct failure mode: a break in a stationary nonlinear
conditional law.  A small deterministic polynomial autoregression is fit on
the first 70% of the guaranteed-clean history and calibrated only on the
remaining 30%, so its online forecast errors and its null residuals are both
out of sample with respect to the fit.

The predictor uses six Gaussian-rank lags, their second/third Hermite terms,
absolute values, and adjacent-lag products.  Eleven centered residual aspects
are accumulated with the same causal prefix/CUSUM/suffix geometry as d3.
Both the offline and streaming implementations deliberately use independent
state paths and must agree to machine precision in the self-test.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri

import detectors3 as d3


ORDER = 6
TRAIN_FRACTION = 0.70
RIDGE = 25.0
WINDOWS = d3.WINDOWS
ASPECTS = (
    "nl_e", "nl_e2", "nl_abse",
    "nl_L1", "nl_L2", "nl_L3", "nl_L4",
    "nl_tail_up", "nl_tail_dn", "nl_ac1", "nl_ac2",
)
BLOCK = 3 + len(WINDOWS)
N_EXTRA = len(ASPECTS) * BLOCK


def feature_names():
    names = []
    for aspect in ASPECTS:
        names.extend((f"{aspect}_prefix", f"{aspect}_up",
                      f"{aspect}_down"))
        names.extend(f"{aspect}_w{window}" for window in WINDOWS)
    return names


def _gaussianize(probability, n_hist):
    epsilon = 0.5 / max(int(n_hist), 1)
    return ndtri(np.clip(np.asarray(probability, dtype=np.float64),
                         epsilon, 1.0 - epsilon))


def _lag_rows(values):
    values = np.asarray(values, dtype=np.float64)
    target = values[ORDER:]
    lags = np.column_stack([
        values[ORDER - lag:values.size - lag]
        for lag in range(1, ORDER + 1)
    ])
    return target, lags


def _basis(lags):
    lags = np.asarray(lags, dtype=np.float64)
    clipped = np.clip(lags, -4.0, 4.0)
    parts = (
        clipped,
        clipped * clipped - 1.0,
        clipped * clipped * clipped - 3.0 * clipped,
        np.abs(clipped),
        clipped[:, :-1] * clipped[:, 1:],
    )
    return np.column_stack(parts)


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
    penalty = RIDGE * np.eye(design.shape[1], dtype=np.float64)
    try:
        coefficient = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.zeros(design.shape[1], dtype=np.float64)
    if not np.all(np.isfinite(coefficient)):
        coefficient[:] = 0.0
    return mean, scale, target_mean, coefficient


def _predict(lags, mean, scale, target_mean, coefficient):
    design = (_basis(lags) - mean) / scale
    return target_mean + design @ coefficient


def _ecdf(sorted_values, values):
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(sorted_values, values, side="left")
    right = np.searchsorted(sorted_values, values, side="right")
    return (left + right) / (2.0 * sorted_values.size)


def _legendre(probability):
    u = np.asarray(probability, dtype=np.float64)
    return (
        2.0 * u - 1.0,
        6.0 * u * u - 6.0 * u + 1.0,
        20.0 * u ** 3 - 30.0 * u * u + 12.0 * u - 1.0,
        70.0 * u ** 4 - 140.0 * u ** 3 + 90.0 * u * u - 20.0 * u + 1.0,
    )


class NonlinearNull:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.hs = d3.Hist3(history)
        gaussian = _gaussianize(self.hs._cdf(history), self.hs.n_hist)
        target, lags = _lag_rows(gaussian)
        n_rows = target.size
        split = int(np.floor(TRAIN_FRACTION * n_rows))
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
        if standardized.size >= 2:
            self.residual_tail = standardized[-2:].copy()
        else:
            self.residual_tail = np.zeros(2, dtype=np.float64)

        probability = _ecdf(self.sorted_residual, residual)
        legendre = _legendre(probability)
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

    def increments(self, standardized, probability, history_mode=False,
                   previous=None):
        e = np.asarray(standardized, dtype=np.float64)
        u = np.asarray(probability, dtype=np.float64)
        legendre = _legendre(u)
        output = {
            "nl_e": e - self.center.get("nl_e", 0.0),
            "nl_e2": e * e - self.center.get("nl_e2", 1.0),
            "nl_abse": np.abs(e) - self.center.get("nl_abse", 0.0),
            "nl_L1": legendre[0] - self.center.get("nl_L1", 0.0),
            "nl_L2": legendre[1] - self.center.get("nl_L2", 0.0),
            "nl_L3": legendre[2] - self.center.get("nl_L3", 0.0),
            "nl_L4": legendre[3] - self.center.get("nl_L4", 0.0),
            "nl_tail_up": (u > 0.95).astype(np.float64)
                          - self.center.get("nl_tail_up", 0.05),
            "nl_tail_dn": (u < 0.05).astype(np.float64)
                          - self.center.get("nl_tail_dn", 0.05),
        }
        ac1 = np.zeros(e.size, dtype=np.float64)
        ac2 = np.zeros(e.size, dtype=np.float64)
        if history_mode:
            if e.size > 1:
                ac1[1:] = e[1:] * e[:-1] - self.center["nl_ac1"]
            if e.size > 2:
                ac2[2:] = e[2:] * e[:-2] - self.center["nl_ac2"]
        else:
            prior = np.asarray(previous, dtype=np.float64)
            full = np.concatenate((prior[-2:], e))
            ac1[:] = full[2:] * full[1:-1] - self.center["nl_ac1"]
            ac2[:] = full[2:] * full[:-2] - self.center["nl_ac2"]
        output["nl_ac1"] = ac1
        output["nl_ac2"] = ac2
        return output


def extra_features(x_hist, x_online):
    null = NonlinearNull(x_hist)
    online = np.asarray(x_online, dtype=np.float64)
    probability = null.hs._cdf(online)
    gaussian = _gaussianize(probability, null.hs.n_hist)
    full = np.concatenate((null.gaussian_tail, gaussian))
    lags = np.column_stack([
        full[ORDER - lag:ORDER - lag + online.size]
        for lag in range(1, ORDER + 1)
    ])
    residual = gaussian - _predict(
        lags, null.mean, null.scale,
        null.target_mean, null.coefficient)
    standardized = (residual - null.residual_mean) / null.residual_sd
    residual_probability = _ecdf(null.sorted_residual, residual)
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


class StreamDetector57:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.null = NonlinearNull(x_hist)
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

    def _from_current_inner(self):
        null = self.null
        gaussian = float(_gaussianize(
            np.asarray([self.inner.u_prev]), null.hs.n_hist)[0])
        prediction = null.predict_one(self.gaussian_lags)
        residual = gaussian - prediction
        standardized = (residual - null.residual_mean) / null.residual_sd
        probability = null.residual_probability(residual)
        legendre = _legendre(np.asarray([probability]))
        increment = {
            "nl_e": standardized - null.center["nl_e"],
            "nl_e2": standardized * standardized - null.center["nl_e2"],
            "nl_abse": abs(standardized) - null.center["nl_abse"],
            "nl_L1": float(legendre[0][0]) - null.center["nl_L1"],
            "nl_L2": float(legendre[1][0]) - null.center["nl_L2"],
            "nl_L3": float(legendre[2][0]) - null.center["nl_L3"],
            "nl_L4": float(legendre[3][0]) - null.center["nl_L4"],
            "nl_tail_up": (1.0 if probability > 0.95 else 0.0)
                          - null.center["nl_tail_up"],
            "nl_tail_dn": (1.0 if probability < 0.05 else 0.0)
                          - null.center["nl_tail_dn"],
            "nl_ac1": standardized * self.residual_lags[0]
                      - null.center["nl_ac1"],
            "nl_ac2": standardized * self.residual_lags[1]
                      - null.center["nl_ac2"],
        }

        output = np.empty(N_EXTRA, dtype=np.float64)
        column = 0
        position = self.t % max(WINDOWS)
        root_count = np.sqrt(self.t + 1.0)
        for aspect in ASPECTS:
            value = float(increment[aspect])
            sd = null.null_sd[aspect]
            self.cumulative[aspect] += value
            cumulative = self.cumulative[aspect]
            self.minimum[aspect] = min(self.minimum[aspect], cumulative)
            self.maximum[aspect] = max(self.maximum[aspect], cumulative)
            output[column] = cumulative / (sd * root_count)
            output[column + 1] = (cumulative - self.minimum[aspect]) / (
                sd * root_count)
            output[column + 2] = (self.maximum[aspect] - cumulative) / (
                sd * root_count)
            column += 3
            ring = self.ring[aspect]
            for window in WINDOWS:
                self.window_sum[aspect][window] += value
                drop = self.t - window
                if drop >= 0:
                    self.window_sum[aspect][window] -= ring[
                        drop % max(WINDOWS)]
                effective = min(self.t + 1, window)
                output[column] = self.window_sum[aspect][window] / (
                    sd * np.sqrt(effective))
                column += 1
            ring[position] = value

        self.gaussian_lags = [gaussian] + self.gaussian_lags[:-1]
        self.residual_lags = [standardized] + self.residual_lags[:-1]
        self.t += 1
        return output

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(57)
    worst = 0.0
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(20, 700))
        total = n_hist + steps
        values = np.zeros(total, dtype=np.float64)
        noise = rng.standard_t(float(rng.uniform(4.0, 12.0)), total)
        before = float(rng.uniform(-0.6, 0.6))
        after = float(np.clip(before + rng.uniform(-0.8, 0.8), -0.9, 0.9))
        change = n_hist + steps // 2
        for index in range(2, total):
            coefficient = after if index >= change else before
            nonlinear = 0.12 * np.tanh(values[index - 2])
            values[index] = (coefficient * values[index - 1]
                             + nonlinear + noise[index])
        history = values[:n_hist]
        online = values[n_hist:]
        offline = extra_features(history, online)
        detector = StreamDetector57(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared_detector = StreamDetector57(history, inner=shared_inner)
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
    detector = StreamDetector57(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000.0
    print(f"detectors57 OK: {N_EXTRA} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6:.0f} us/step")
