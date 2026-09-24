"""History-fitted Rosenblatt channels on the missing width-five scale."""
from __future__ import annotations

import numpy as np

from models.model57.whitening import (
    CHANNELS,
    _empirical_normal,
    _fit_ar,
    _raw_channels,
    _residuals,
    _robust_scale,
    _variance_paths,
)

WINDOW = 5
FEATURE_NAMES = tuple(f"white_{channel}_w5" for channel in CHANNELS) + (
    "white_online_log2_age_plus1",
)


def short_whitening_features(history, online):
    """Return eight causal features for every observed online prefix."""
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if (h.ndim != 1 or o.ndim != 1 or len(h) < 64 or
            not np.isfinite(h).all() or not np.isfinite(o).all()):
        raise ValueError("Invalid series for Model88 whitening")
    center, scale = _robust_scale(h)
    hz = np.clip((h - center) / scale, -20, 20)
    oz = np.clip((o - center) / scale, -20, 20)
    order, intercept, phi = _fit_ar(hz)
    eh, eo = _residuals(hz, oz, order, intercept, phi)
    rh, ro = _variance_paths(eh, eo)
    zh = _empirical_normal(rh, rh)
    zo = _empirical_normal(rh, ro)
    historical = _raw_channels(zh, np.r_[0.0, zh[:-1]])
    current = _raw_channels(zo, np.r_[zh[-1], zo[:-1]])
    reference = historical[min(16, len(historical) - 1):]
    channel_center = np.mean(reference, axis=0)
    channel_scale = np.maximum(np.std(reference, axis=0), 0.15)
    adjusted = np.clip((current - channel_center) / channel_scale, -8, 8)

    end = np.arange(1, len(o) + 1)
    start = np.maximum(0, end - WINDOW)
    prefix = np.vstack((np.zeros((1, len(CHANNELS))), np.cumsum(adjusted, axis=0)))
    block = np.clip((prefix[end] - prefix[start]) /
                    np.sqrt(end - start)[:, None], -20, 20)
    age = np.log2(end.astype(np.float64) + 1.0)
    matrix = np.ascontiguousarray(np.column_stack((block, age)), dtype=np.float32)
    if matrix.shape != (len(o), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid Model88 short-whitening features")
    return matrix


def prefix_parity(history, online):
    full = short_whitening_features(history, online)
    for cut in (1, 2, 5, 8, 16, 64, 128, 256):
        if cut <= len(online):
            np.testing.assert_array_equal(
                short_whitening_features(history, online[:cut])[-1], full[cut - 1])
    return full
