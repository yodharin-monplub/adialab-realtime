"""Causal Sup-F-like step and piecewise-regression breakpoint scans.

At each online step the detector scans fixed candidate ages inside several
fully observed suffixes.  It never estimates a breakpoint using future rows.
All scan outputs are standardized against identical scans of rolling windows
from the clean historical segment.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sequential_utils import (
    aggregate_maxima,
    fit_location_scale,
    last_maxima,
    standardize,
)


WINDOWS = (32, 64, 128, 256)
SPLIT_FRACTIONS = (0.25, 0.50, 0.75)
METRICS = (
    "step_signed",
    "step_abs",
    "step_fraction",
    "scale_signed",
    "piecewise_gain",
    "slope_change",
)
MAX_WINDOW = max(WINDOWS)


def feature_names() -> list[str]:
    names = [f"{metric}_w{window}" for window in WINDOWS for metric in METRICS]
    for metric in METRICS:
        names.extend((f"{metric}_maxabs", f"{metric}_signedmax"))
    return names


N_FEATURES = len(feature_names())


def _linear_sse(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return intercept+slope SSE and slope for equally spaced rows."""

    width = matrix.shape[1]
    centered_x = np.arange(width, dtype=np.float64) - 0.5 * (width - 1)
    ssx = max(float(centered_x @ centered_x), 1e-12)
    centered_y = matrix - matrix.mean(axis=1, keepdims=True)
    covariance = centered_y @ centered_x
    slope = covariance / ssx
    sse = np.sum(centered_y * centered_y, axis=1) - covariance * covariance / ssx
    return np.maximum(sse, 0.0), slope


def _scan_windows(values: np.ndarray, window: int) -> np.ndarray:
    """Evaluate fixed candidate break locations in every complete window."""

    series = np.asarray(values, dtype=np.float64)
    if series.size < window:
        return np.empty((0, len(METRICS)), dtype=np.float64)
    matrix = sliding_window_view(series, window)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    total_sse = np.sum(centered * centered, axis=1)
    total_scale = np.sqrt(np.maximum(total_sse / window, 1e-12))
    global_line_sse, _ = _linear_sse(matrix)

    step_values = []
    scale_values = []
    constant_gains = []
    line_gains = []
    slope_changes = []
    for fraction in SPLIT_FRACTIONS:
        split = int(round(window * fraction))
        left = matrix[:, :split]
        right = matrix[:, split:]
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        left_sse = np.sum(left_centered * left_centered, axis=1)
        right_sse = np.sum(right_centered * right_centered, axis=1)
        contrast = np.sqrt(split * (window - split) / window) * (
            right.mean(axis=1) - left.mean(axis=1)
        ) / total_scale
        step_values.append(contrast)
        constant_gains.append(
            np.maximum(total_sse - left_sse - right_sse, 0.0)
            / np.maximum(total_sse, 1e-12)
        )
        left_line_sse, left_slope = _linear_sse(left)
        right_line_sse, right_slope = _linear_sse(right)
        line_gains.append(
            np.maximum(global_line_sse - left_line_sse - right_line_sse, 0.0)
            / np.maximum(global_line_sse, 1e-12)
        )
        slope_changes.append(
            window * (right_slope - left_slope) / np.maximum(total_scale, 1e-12)
        )
        left_variance = np.maximum(left_sse / split, 1e-12)
        right_variance = np.maximum(right_sse / (window - split), 1e-12)
        scale_values.append(np.log(right_variance / left_variance))

    step = np.column_stack(step_values)
    scale = np.column_stack(scale_values)
    gain = np.column_stack(constant_gains)
    line_gain = np.column_stack(line_gains)
    slope = np.column_stack(slope_changes)
    row = np.arange(matrix.shape[0])
    step_index = np.argmax(np.abs(step), axis=1)
    scale_index = np.argmax(np.abs(scale), axis=1)
    slope_index = np.argmax(np.abs(slope), axis=1)
    signed_step = step[row, step_index]
    split_fraction = np.asarray(SPLIT_FRACTIONS)[step_index] - 0.5
    return np.column_stack(
        (
            signed_step,
            np.abs(signed_step),
            split_fraction,
            scale[row, scale_index],
            np.max(np.maximum(gain, line_gain), axis=1),
            slope[row, slope_index],
        )
    )


class HistoryNull62:
    """History-only calibration for every scan horizon."""

    def __init__(self, historical: np.ndarray):
        history = np.asarray(historical, dtype=np.float64)
        if history.ndim != 1 or history.size < MAX_WINDOW + 8:
            raise ValueError("detectors62 needs at least 264 historical values")
        self.calibration = {
            window: fit_location_scale(_scan_windows(history, window))
            for window in WINDOWS
        }

    def score(self, window: int, raw: np.ndarray) -> np.ndarray:
        center, scale = self.calibration[window]
        return standardize(raw, center, scale)


def compute_features_offline(
    historical: np.ndarray, online: np.ndarray
) -> np.ndarray:
    """Vectorized causal scan features for all online endpoints."""

    null = HistoryNull62(historical)
    values = np.asarray(online, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    columns: list[np.ndarray] = []
    by_metric = {metric: [] for metric in METRICS}
    for window in WINDOWS:
        score = np.zeros((values.size, len(METRICS)), dtype=np.float64)
        raw = _scan_windows(values, window)
        if raw.shape[0]:
            score[window - 1 :] = null.score(window, raw)
        for index, metric in enumerate(METRICS):
            columns.append(score[:, index])
            by_metric[metric].append(score[:, index])
    columns.extend(aggregate_maxima(by_metric, METRICS))
    result = np.column_stack(columns)
    if result.shape != (values.size, N_FEATURES):
        raise AssertionError("detectors62 emitted an invalid feature shape")
    return result


class StreamDetector62:
    """Bounded-memory streaming implementation of the fixed scan grid."""

    def __init__(self, historical: np.ndarray):
        self.null = HistoryNull62(historical)
        self.values: list[float] = []

    def step(self, value: float) -> np.ndarray:
        self.values.append(float(value))
        if len(self.values) > MAX_WINDOW:
            del self.values[0]
        output: list[float] = []
        by_metric = {metric: [] for metric in METRICS}
        for window in WINDOWS:
            if len(self.values) < window:
                score = np.zeros(len(METRICS), dtype=np.float64)
            else:
                raw = _scan_windows(np.asarray(self.values[-window:]), window)[0]
                score = self.null.score(window, raw)
            output.extend(float(item) for item in score)
            for metric, item in zip(METRICS, score, strict=True):
                by_metric[metric].append(float(item))
        output.extend(last_maxima(by_metric, METRICS))
        result = np.asarray(output, dtype=np.float64)
        if result.size != N_FEATURES:
            raise AssertionError("detectors62 streaming contract changed")
        return result


if __name__ == "__main__":
    rng = np.random.default_rng(6201)
    history = rng.normal(size=900)
    online = rng.normal(size=170)
    online[88:] += np.linspace(0.0, 1.6, online.size - 88)
    offline = compute_features_offline(history, online)
    detector = StreamDetector62(history)
    streamed = np.vstack([detector.step(value) for value in online])
    error = float(np.max(np.abs(offline - streamed)))
    assert error < 1e-9 and np.all(np.isfinite(offline))
    print(f"OK: detectors62 stream == offline; {N_FEATURES} features; error={error:.3g}")
