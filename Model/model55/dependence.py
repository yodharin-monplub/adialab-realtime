"""Causal, history-referenced higher-lag correlation contrasts."""
from __future__ import annotations

import numpy as np

LAGS = (2, 4, 8, 16)
WINDOWS = (64, 256, 0)
STREAMS = ("raw", "diff")
FEATURE_NAMES = tuple(
    f"{stream}_lag{lag}_{'prefix' if window == 0 else f'w{window}'}"
    for stream in STREAMS for window in WINDOWS for lag in LAGS
) + tuple(
    f"{stream}_maxabs_{'prefix' if window == 0 else f'w{window}'}"
    for stream in STREAMS for window in WINDOWS
)


def _standardize(history, online):
    center = float(np.median(history))
    scale = float(1.4826 * np.median(np.abs(history - center)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = max(float(np.std(history)), 1e-8)
    h = np.clip((history - center) / scale, -8, 8)
    o = np.clip((online - center) / scale, -8, 8)
    return h, o


def _correlation(a, b):
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.mean((a - ma) ** 2)), float(np.mean((b - mb) ** 2))
    if min(va, vb) < 1e-10:
        return 0.
    return float(np.clip(np.mean((a - ma) * (b - mb)) / np.sqrt(va * vb), -.98, .98))


def _contrast(history, online, lag, window):
    result = np.zeros(len(online), dtype=np.float32)
    if len(history) < lag + 32 or len(online) <= lag:
        return result
    baseline = np.arctanh(_correlation(history[lag:], history[:-lag]))
    a, b = online[lag:].astype(np.float64), online[:-lag].astype(np.float64)
    n_pairs = len(a)
    end = np.arange(1, n_pairs + 1, dtype=np.int64)
    start = np.zeros_like(end) if window == 0 else np.maximum(0, end - window)
    count = (end - start).astype(np.float64)

    def sums(values):
        prefix = np.r_[0., np.cumsum(values, dtype=np.float64)]
        return prefix[end] - prefix[start]

    mean_a, mean_b = sums(a) / count, sums(b) / count
    covariance = sums(a * b) / count - mean_a * mean_b
    variance_a = np.maximum(sums(a * a) / count - mean_a * mean_a, 0.)
    variance_b = np.maximum(sums(b * b) / count - mean_b * mean_b, 0.)
    correlation = covariance / np.sqrt(variance_a * variance_b + 1e-8)
    correlation = np.clip(correlation, -.98, .98)
    statistic = np.clip(np.sqrt(count) * (np.arctanh(correlation) - baseline), -20, 20)
    statistic[count < 16] = 0.
    result[lag:] = statistic.astype(np.float32)
    return result


def dependence_features(history, online):
    """Return one feature row per online observation, never using its future."""
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if h.ndim != 1 or o.ndim != 1 or len(h) < 48 or not np.isfinite(h).all() or not np.isfinite(o).all():
        raise ValueError("Invalid series for dependence features")
    streams = {
        "raw": _standardize(h, o),
        "diff": _standardize(np.diff(h), np.diff(np.r_[h[-1], o])),
    }
    columns = []
    maxima = []
    for stream in STREAMS:
        clean, current = streams[stream]
        for window in WINDOWS:
            block = [_contrast(clean, current, lag, window) for lag in LAGS]
            columns.extend(block)
            maxima.append(np.max(np.abs(np.stack(block)), axis=0).astype(np.float32))
    columns.extend(maxima)
    matrix = np.column_stack(columns).astype(np.float32)
    if matrix.shape != (len(o), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid dependence feature matrix")
    return matrix
