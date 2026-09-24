"""Causal AR(p) mean-shift likelihood features with exact boundary weights.

For a candidate mean change at tau, the null AR residual at age j changes by
delta * (1 - sum(rho[:min(p, j)])).  The normalized score below is therefore
the signed square root of the known-prechange-mean GLR for that suffix.  A
small fixed suffix grid approximates the full FOCuS candidate scan while
remaining cheap enough for competition streaming.
"""
from __future__ import annotations

import numpy as np


ORDERS = (1, 2, 5, 10)
WINDOWS = (8, 16, 32, 64, 128, 256)
MAX_ORDER = max(ORDERS)
MAX_WINDOW = max(WINDOWS)


def feature_names():
    names = []
    for order in ORDERS:
        for window in WINDOWS:
            names.extend((f"arpfocus_p{order}_signed_w{window}",
                          f"arpfocus_p{order}_stat_w{window}"))
        names.extend((f"arpfocus_p{order}_max",
                      f"arpfocus_p{order}_signed_at_max",
                      f"arpfocus_p{order}_log2_age"))
    return names


N_FEATURES = len(feature_names())


def _yule_walker(values, order, ridge=1e-3):
    x = np.asarray(values, dtype=np.float64)
    centered = x - x.mean()
    covariance = np.asarray([
        np.mean(centered[lag:] * centered[:-lag]) if lag else np.mean(centered * centered)
        for lag in range(order + 1)
    ])
    matrix = covariance[np.abs(np.subtract.outer(np.arange(order), np.arange(order)))]
    matrix = matrix + np.eye(order) * ridge * max(float(covariance[0]), 1e-12)
    try:
        coefficients = np.linalg.solve(matrix, covariance[1:])
    except np.linalg.LinAlgError:
        coefficients = np.zeros(order)
    if not np.isfinite(coefficients).all() or np.abs(coefficients).sum() > 2.5:
        coefficients = np.zeros(order)
    return coefficients.astype(np.float64)


def _residuals(values, coefficients, context):
    x = np.asarray(values, dtype=np.float64)
    order = len(coefficients)
    full = np.concatenate((np.asarray(context, dtype=np.float64)[-order:], x))
    residual = x.copy()
    for lag in range(1, order + 1):
        residual -= coefficients[lag - 1] * full[order - lag:order - lag + len(x)]
    return residual


class ARNull:
    def __init__(self, historical):
        history = np.asarray(historical, dtype=np.float64)
        if history.ndim != 1 or history.size < 12 * MAX_ORDER or not np.isfinite(history).all():
            raise ValueError("AR-Focus history must be a sufficiently long finite vector")
        self.mean = float(history.mean())
        scale = float(history.std())
        self.scale = max(scale, 1e-12)
        z = (history - self.mean) / self.scale
        self.tail = z[-MAX_ORDER:].copy()
        self.coefficients = {}
        self.residual_scale = {}
        self.boundary = {}
        for order in ORDERS:
            phi = _yule_walker(z, order)
            residual = _residuals(z, phi, np.zeros(order))[order:]
            residual_scale = max(float(residual.std()), 1e-8)
            weights = np.asarray([
                1.0 - phi[:min(order, age)].sum() for age in range(order)
            ], dtype=np.float64)
            self.coefficients[order] = phi
            self.residual_scale[order] = residual_scale
            self.boundary[order] = weights


def _offline_order(null, z, order):
    phi = null.coefficients[order]
    residual = _residuals(z, phi, null.tail) / null.residual_scale[order]
    cumulative = np.concatenate(([0.0], np.cumsum(residual)))
    count = np.arange(1, len(z) + 1, dtype=np.int64)
    constant = 1.0 - phi.sum()
    boundary = null.boundary[order]
    correction = boundary - constant
    signed_columns = []
    statistic_columns = []
    ages = []
    for window in WINDOWS:
        length = np.minimum(count, window)
        start = count - length
        numerator = constant * (cumulative[count] - cumulative[start])
        denominator = constant * constant * length.astype(np.float64)
        for age in range(order):
            valid = length > age
            if np.any(valid):
                positions = start[valid] + age
                numerator[valid] += correction[age] * residual[positions]
                denominator[valid] += boundary[age] ** 2 - constant ** 2
        signed = numerator / np.sqrt(np.maximum(denominator, 1e-10))
        signed_columns.append(signed)
        statistic_columns.append(signed * signed)
        ages.append(length)
    statistic = np.column_stack(statistic_columns)
    signed = np.column_stack(signed_columns)
    best = np.argmax(statistic, axis=1)
    row = np.arange(len(z))
    columns = []
    for index in range(len(WINDOWS)):
        columns.extend((signed[:, index], statistic[:, index]))
    columns.extend((statistic[row, best], signed[row, best],
                    np.log2(np.column_stack(ages)[row, best].astype(np.float64))))
    return np.column_stack(columns)


def compute_features_offline(historical, online):
    null = ARNull(historical)
    values = np.asarray(online, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("AR-Focus online input must be a finite vector")
    if values.size == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    z = (values - null.mean) / null.scale
    result = np.column_stack([_offline_order(null, z, order) for order in ORDERS])
    if result.shape != (len(values), N_FEATURES) or not np.isfinite(result).all():
        raise ValueError("Invalid AR-Focus feature output")
    return result


class StreamDetector:
    def __init__(self, historical):
        self.null = ARNull(historical)
        self.lags = list(self.null.tail[::-1])
        self.time = 0
        self.rings = {order: np.zeros(MAX_WINDOW) for order in ORDERS}
        self.sums = {order: {window: 0.0 for window in WINDOWS} for order in ORDERS}

    def step(self, value):
        z = (float(value) - self.null.mean) / self.null.scale
        time = self.time
        count = time + 1
        position = time % MAX_WINDOW
        output = []
        by_order = {}
        for order in ORDERS:
            phi = self.null.coefficients[order]
            residual = (z - float(np.dot(phi, self.lags[:order]))) / self.null.residual_scale[order]
            ring = self.rings[order]
            signed_values = []
            statistics = []
            ages = []
            for window in WINDOWS:
                self.sums[order][window] += residual
                if time >= window:
                    self.sums[order][window] -= ring[(time - window) % MAX_WINDOW]
            ring[position] = residual
            constant = 1.0 - phi.sum()
            boundary = self.null.boundary[order]
            for window in WINDOWS:
                length = min(count, window)
                start = count - length
                numerator = constant * self.sums[order][window]
                denominator = constant * constant * length
                for age in range(min(order, length)):
                    item = ring[(start + age) % MAX_WINDOW]
                    numerator += (boundary[age] - constant) * item
                    denominator += boundary[age] ** 2 - constant ** 2
                signed = numerator / np.sqrt(max(denominator, 1e-10))
                statistic = signed * signed
                output.extend((signed, statistic))
                signed_values.append(signed)
                statistics.append(statistic)
                ages.append(length)
            best = int(np.argmax(statistics))
            output.extend((statistics[best], signed_values[best], np.log2(float(ages[best]))))
            by_order[order] = residual
        self.lags = [z] + self.lags[:MAX_ORDER - 1]
        self.time = count
        result = np.asarray(output, dtype=np.float64)
        if result.shape != (N_FEATURES,) or not np.isfinite(result).all():
            raise ValueError("Invalid streaming AR-Focus features")
        return result


if __name__ == "__main__":
    rng = np.random.default_rng(3601)
    for trial in range(8):
        history_length = int(rng.integers(500, 2500))
        online_length = int(rng.integers(30, 500))
        coefficient = float(rng.uniform(-.75, .75))
        noise = rng.normal(size=history_length + online_length)
        values = np.empty_like(noise)
        values[0] = noise[0]
        for index in range(1, len(values)):
            values[index] = coefficient * values[index - 1] + noise[index]
        if trial % 2 == 0:
            values[history_length + online_length // 2:] += .7
        history, online = values[:history_length], values[history_length:]
        offline = compute_features_offline(history, online)
        detector = StreamDetector(history)
        streamed = np.vstack([detector.step(item) for item in online])
        np.testing.assert_allclose(offline, streamed, rtol=1e-10, atol=1e-10)
    assert N_FEATURES == 60 and len(feature_names()) == N_FEATURES
    print("AR-Focus feature parity passed")
