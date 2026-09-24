"""Causal higher-lag innovation-dependence evidence on history-fitted whitening."""
from __future__ import annotations

import numpy as np

from models.model57.whitening import (_empirical_normal, _fit_ar, _residuals,
    _robust_scale, _variance_paths)

LAGS = (2, 4, 8, 16)
WINDOWS = (16, 64, 256, 0)
CHANNELS = tuple(f"{kind}_lag{lag}" for kind in ("signed", "magnitude") for lag in LAGS)
FEATURE_NAMES = tuple(
    f"white_dep_{channel}_{'prefix' if window == 0 else f'w{window}'}"
    for window in WINDOWS for channel in CHANNELS
) + tuple(f"white_dep_{channel}_maxabs" for channel in CHANNELS)


def innovation_dependence_features(history, online):
    """Return 40 prefix-invariant columns for each observed online point."""
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if h.ndim != 1 or o.ndim != 1 or len(h) < 64 or not np.isfinite(h).all() or not np.isfinite(o).all():
        raise ValueError("Invalid series for Model66 innovation dependence")
    if len(o) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    center, scale = _robust_scale(h)
    hz = np.clip((h - center) / scale, -20, 20)
    oz = np.clip((o - center) / scale, -20, 20)
    order, intercept, phi = _fit_ar(hz)
    eh, eo = _residuals(hz, oz, order, intercept, phi)
    rh, ro = _variance_paths(eh, eo)
    clean = _empirical_normal(rh, rh)
    current = _empirical_normal(rh, ro)
    historical_channels, online_channels = [], []
    for kind in ("signed", "magnitude"):
        for lag in LAGS:
            reference = clean[lag:] * clean[:-lag]
            previous = np.r_[clean[-lag:], current][:-lag]
            observed = current * previous
            if kind == "magnitude":
                reference, observed = np.abs(reference), np.abs(observed)
            historical_channels.append(reference)
            online_channels.append(observed)
    adjusted = []
    for reference, observed in zip(historical_channels, online_channels):
        reference = reference[min(16, len(reference) - 1):]
        mean = float(np.mean(reference))
        sd = max(float(np.std(reference)), 0.15)
        adjusted.append(np.clip((observed - mean) / sd, -8, 8))
    channels = np.column_stack(adjusted)
    prefix = np.vstack((np.zeros((1, len(CHANNELS))), np.cumsum(channels, axis=0)))
    end = np.arange(1, len(o) + 1)
    blocks = []
    for window in WINDOWS:
        start = np.zeros_like(end) if window == 0 else np.maximum(0, end - window)
        count = end - start
        blocks.append(np.clip((prefix[end] - prefix[start]) / np.sqrt(count)[:, None], -20, 20))
    maximum = np.max(np.abs(np.stack(blocks)), axis=0)
    matrix = np.column_stack((*blocks, maximum))
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if matrix.shape != (len(o), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid Model66 innovation-dependence features")
    return matrix
