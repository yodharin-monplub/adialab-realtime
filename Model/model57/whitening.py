"""History-only AR/EWMA/Rosenblatt whitening and causal innovation features."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.special import ndtri

AR_ORDERS = (0, 1, 2, 4, 8, 12)
EWMA_LAMBDA = .97
WINDOWS = (16, 32, 64, 128, 256, 512, 0)
CHANNELS = ("location", "squared", "absolute", "lag1", "tail_up",
            "tail_down", "abs_change")
CONTEXT = ("ar_order", "phi1", "residual_ratio", "vol_cluster", "log_kurtosis")
FEATURE_NAMES = tuple(
    f"white_{channel}_{'prefix' if window == 0 else f'w{window}'}"
    for window in WINDOWS for channel in CHANNELS
) + tuple(f"white_{channel}_maxabs" for channel in CHANNELS) + tuple(
    f"white_hist_{name}" for name in CONTEXT)


def _robust_scale(values):
    x = np.asarray(values, dtype=np.float64)
    center = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - center)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = max(float(np.std(x)), 1e-8)
    return center, scale


def _fit_ar(history):
    h = np.asarray(history[-min(len(history), 2048):], dtype=np.float64)
    if len(h) < 64:
        raise ValueError("Insufficient clean history for AR order selection")
    target = h[max(AR_ORDERS):]
    length = len(target)
    best = None
    for order in AR_ORDERS:
        if order:
            lagged = np.column_stack(
                [h[max(AR_ORDERS) - lag:len(h) - lag] for lag in range(1, order + 1)])
            design = np.column_stack((np.ones(length), lagged))
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        else:
            coefficients = np.asarray([float(np.mean(target))])
        residual = target - coefficients[0]
        if order:
            residual -= lagged @ coefficients[1:]
        mse = max(float(np.mean(residual * residual)), 1e-8)
        bic = length * np.log(mse) + (order + 1) * np.log(length)
        if best is None or bic < best[0]:
            best = (bic, order, coefficients.copy())
    _, order, coefficients = best
    coefficients[1:] = np.clip(coefficients[1:], -3, 3)
    return int(order), float(coefficients[0]), coefficients[1:]


def _residuals(history, online, order, intercept, phi):
    h, o = np.asarray(history, dtype=np.float64), np.asarray(online, dtype=np.float64)
    if order == 0:
        return h - intercept, o - intercept
    windows = sliding_window_view(h, order + 1)
    history_residual = windows[:, order] - intercept - windows[:, :order][:, ::-1] @ phi
    combined = np.r_[h[-order:], o]
    predictors = sliding_window_view(combined, order)[:len(o), ::-1]
    online_residual = o - intercept - predictors @ phi
    return history_residual, online_residual


def _variance_paths(history_residual, online_residual):
    h = np.clip(np.asarray(history_residual, dtype=np.float64), -20, 20)
    o = np.clip(np.asarray(online_residual, dtype=np.float64), -20, 20)
    variance0 = max(float(np.mean(h * h)), 1e-4)
    historical_forcing = (1 - EWMA_LAMBDA) * np.r_[variance0, h[:-1] * h[:-1]]
    historical_variance = lfilter([1.], [1., -EWMA_LAMBDA], historical_forcing,
        zi=np.asarray([EWMA_LAMBDA * variance0]))[0]
    online_forcing = (1 - EWMA_LAMBDA) * np.r_[h[-1] * h[-1], o[:-1] * o[:-1]]
    online_variance = lfilter([1.], [1., -EWMA_LAMBDA], online_forcing,
        zi=np.asarray([EWMA_LAMBDA * historical_variance[-1]]))[0]
    historical_variance = np.clip(historical_variance, 1e-4, 100.)
    online_variance = np.clip(online_variance, 1e-4, 100.)
    return h / np.sqrt(historical_variance), o / np.sqrt(online_variance)


def _empirical_normal(reference, values):
    sorted_ref = np.sort(np.asarray(reference, dtype=np.float64))
    query = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(sorted_ref, query, side="left")
    right = np.searchsorted(sorted_ref, query, side="right")
    probability = (left + right + 1.) / (2. * (len(sorted_ref) + 1.))
    return np.clip(ndtri(probability), -3.75, 3.75)


def _raw_channels(values, previous):
    z = np.asarray(values, dtype=np.float64)
    p = np.asarray(previous, dtype=np.float64)
    return np.column_stack((z, z * z, np.abs(z), z * p,
        (z > 2.5).astype(np.float64), (z < -2.5).astype(np.float64),
        np.abs(z - p)))


def whitening_features(history, online):
    """Return 61 prefix-invariant features per online observation."""
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if h.ndim != 1 or o.ndim != 1 or len(h) < 64 or not np.isfinite(h).all() or not np.isfinite(o).all():
        raise ValueError("Invalid series for Model57 whitening")
    center, scale = _robust_scale(h)
    hz = np.clip((h - center) / scale, -20, 20)
    oz = np.clip((o - center) / scale, -20, 20)
    order, intercept, phi = _fit_ar(hz)
    eh, eo = _residuals(hz, oz, order, intercept, phi)
    rh, ro = _variance_paths(eh, eo)
    zh = _empirical_normal(rh, rh)
    zo = _empirical_normal(rh, ro)
    historical = _raw_channels(zh, np.r_[0., zh[:-1]])
    current = _raw_channels(zo, np.r_[zh[-1], zo[:-1]])
    reference = historical[min(16, len(historical) - 1):]
    channel_center = np.mean(reference, axis=0)
    channel_scale = np.maximum(np.std(reference, axis=0), .15)
    adjusted = np.clip((current - channel_center) / channel_scale, -8, 8)
    length = len(o)
    prefix = np.vstack((np.zeros((1, len(CHANNELS))), np.cumsum(adjusted, axis=0)))
    end = np.arange(1, length + 1)
    blocks = []
    for window in WINDOWS:
        start = np.zeros_like(end) if window == 0 else np.maximum(0, end - window)
        count = end - start
        blocks.append(np.clip((prefix[end] - prefix[start]) / np.sqrt(count)[:, None], -20, 20))
    maximum = np.max(np.abs(np.stack(blocks)), axis=0)
    phi1 = float(phi[0]) if order else 0.
    squared = np.clip(eh * eh, 0, 100)
    vol_cluster = 0. if len(squared) < 3 else float(np.corrcoef(squared[1:], squared[:-1])[0, 1])
    if not np.isfinite(vol_cluster):
        vol_cluster = 0.
    residual_scale = max(float(np.std(eh)), 1e-8)
    context = np.asarray((order / 12., phi1,
        np.log(residual_scale / max(float(np.std(hz)), 1e-8)),
        np.clip(vol_cluster, -1, 1),
        np.log(max(float(np.mean(np.clip(eh / residual_scale, -10, 10) ** 4)), 1.))),
        dtype=np.float64)
    matrix = np.column_stack((*blocks, maximum, np.tile(context, (length, 1))))
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if matrix.shape != (length, len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid Model57 whitening features")
    return matrix
