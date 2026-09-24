"""Robust time trends and Fisher-transformed dependence changes.

Frozen historical ranks make the time correlations resistant to extreme
values.  Lag correlations and PACF-like evidence are Fisher transformed and
then calibrated on rolling clean-history windows.  Every online window ends
at the current observation.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sequential_utils import (
    aggregate_maxima,
    fit_location_scale,
    frozen_midrank,
    last_maxima,
    standardize,
)


WINDOWS = (16, 32, 64, 128, 256)
METRICS = (
    "spearman_level",
    "spearman_magnitude",
    "acf1_fisher",
    "acf2_fisher",
    "acf5_fisher",
    "pacf2_fisher",
    "magnitude_slope",
)
MAX_WINDOW = max(WINDOWS)


def feature_names() -> list[str]:
    names = [f"{metric}_w{window}" for window in WINDOWS for metric in METRICS]
    for metric in METRICS:
        names.extend((f"{metric}_maxabs", f"{metric}_signedmax"))
    return names


N_FEATURES = len(feature_names())


def _correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if b.ndim == 1:
        b = np.broadcast_to(b, a.shape)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    numerator = np.sum(a * b, axis=1)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def _fisher(correlation: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(correlation, -0.995, 0.995))


def _raw_windows(
    z: np.ndarray, rank: np.ndarray, magnitude_rank: np.ndarray, window: int
) -> np.ndarray:
    if z.size < window:
        return np.empty((0, len(METRICS)), dtype=np.float64)
    z_window = sliding_window_view(np.asarray(z, dtype=np.float64), window)
    rank_window = sliding_window_view(np.asarray(rank, dtype=np.float64), window)
    magnitude_window = sliding_window_view(
        np.asarray(magnitude_rank, dtype=np.float64), window
    )
    time = np.arange(window, dtype=np.float64)
    level_trend = _correlation(rank_window, time)
    magnitude_trend = _correlation(magnitude_window, time)
    acf = {
        lag: _correlation(z_window[:, :-lag], z_window[:, lag:])
        for lag in (1, 2, 5)
    }
    pacf2 = (acf[2] - acf[1] * acf[1]) / np.maximum(
        1.0 - acf[1] * acf[1], 1e-6
    )
    raw_magnitude = _correlation(np.abs(z_window), time)
    return np.column_stack(
        (
            level_trend,
            magnitude_trend,
            _fisher(acf[1]),
            _fisher(acf[2]),
            _fisher(acf[5]),
            _fisher(pacf2),
            raw_magnitude,
        )
    )


class HistoryNull64:
    """Frozen rank transforms and rolling dependence calibrations."""

    def __init__(self, historical: np.ndarray):
        history = np.asarray(historical, dtype=np.float64)
        if history.ndim != 1 or history.size < MAX_WINDOW + 8:
            raise ValueError("detectors64 needs at least 264 historical values")
        self.mean = float(history.mean())
        self.scale = max(float(history.std()), 1e-12)
        self.sorted_history = np.sort(history)
        historical_z = (history - self.mean) / self.scale
        self.sorted_magnitude = np.sort(np.abs(historical_z))
        z, rank, magnitude_rank = self.transform_array(history)
        self.calibration = {
            window: fit_location_scale(_raw_windows(z, rank, magnitude_rank, window))
            for window in WINDOWS
        }

    def transform_array(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.asarray(values, dtype=np.float64)
        z = (query - self.mean) / self.scale
        rank = frozen_midrank(self.sorted_history, query)
        magnitude_rank = frozen_midrank(self.sorted_magnitude, np.abs(z))
        return z, rank, magnitude_rank

    def score(self, window: int, raw: np.ndarray) -> np.ndarray:
        center, scale = self.calibration[window]
        return standardize(raw, center, scale)


def compute_features_offline(
    historical: np.ndarray, online: np.ndarray
) -> np.ndarray:
    """Vectorized robust trend/dependence features for online suffixes."""

    null = HistoryNull64(historical)
    values = np.asarray(online, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    z, rank, magnitude_rank = null.transform_array(values)
    columns: list[np.ndarray] = []
    by_metric = {metric: [] for metric in METRICS}
    for window in WINDOWS:
        score = np.zeros((values.size, len(METRICS)), dtype=np.float64)
        raw = _raw_windows(z, rank, magnitude_rank, window)
        if raw.shape[0]:
            score[window - 1 :] = null.score(window, raw)
        for index, metric in enumerate(METRICS):
            columns.append(score[:, index])
            by_metric[metric].append(score[:, index])
    columns.extend(aggregate_maxima(by_metric, METRICS))
    result = np.column_stack(columns)
    if result.shape != (values.size, N_FEATURES):
        raise AssertionError("detectors64 emitted an invalid feature shape")
    return result


class StreamDetector64:
    """Bounded suffix-buffer streaming counterpart."""

    def __init__(self, historical: np.ndarray):
        self.null = HistoryNull64(historical)
        self.z: list[float] = []
        self.rank: list[float] = []
        self.magnitude_rank: list[float] = []

    def step(self, value: float) -> np.ndarray:
        z, rank, magnitude = self.null.transform_array(np.asarray([value]))
        self.z.append(float(z[0]))
        self.rank.append(float(rank[0]))
        self.magnitude_rank.append(float(magnitude[0]))
        if len(self.z) > MAX_WINDOW:
            del self.z[0]
            del self.rank[0]
            del self.magnitude_rank[0]
        output: list[float] = []
        by_metric = {metric: [] for metric in METRICS}
        for window in WINDOWS:
            if len(self.z) < window:
                score = np.zeros(len(METRICS), dtype=np.float64)
            else:
                raw = _raw_windows(
                    np.asarray(self.z[-window:]),
                    np.asarray(self.rank[-window:]),
                    np.asarray(self.magnitude_rank[-window:]),
                    window,
                )[0]
                score = self.null.score(window, raw)
            output.extend(float(item) for item in score)
            for metric, item in zip(METRICS, score, strict=True):
                by_metric[metric].append(float(item))
        output.extend(last_maxima(by_metric, METRICS))
        result = np.asarray(output, dtype=np.float64)
        if result.size != N_FEATURES:
            raise AssertionError("detectors64 streaming contract changed")
        return result


if __name__ == "__main__":
    rng = np.random.default_rng(6401)
    history = rng.standard_t(7, 900)
    online = rng.standard_t(7, 180)
    for index in range(91, online.size):
        online[index] += 0.8 * online[index - 1]
    offline = compute_features_offline(history, online)
    detector = StreamDetector64(history)
    streamed = np.vstack([detector.step(value) for value in online])
    error = float(np.max(np.abs(offline - streamed)))
    assert error < 1e-9 and np.all(np.isfinite(offline))
    print(f"OK: detectors64 stream == offline; {N_FEATURES} features; error={error:.3g}")
