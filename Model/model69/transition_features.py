"""Causal nonlinear transition-change statistics from history-defined rank bins."""
from __future__ import annotations

import numpy as np

WINDOWS = (64, 256, 0)
STATISTICS = ("glr", "glr_centered", "cell_max", "persistence", "upward", "downward")
FEATURE_NAMES = tuple(
    f"rank_transition_{stat}_{'prefix' if window == 0 else f'w{window}'}"
    for window in WINDOWS for stat in STATISTICS
)


def transition_features(history, online):
    """Compare observed four-state Markov transitions to the clean history.

    The quantile boundaries and conditional transition probabilities are fitted
    only on history. At online time t, every output uses observations <= t.
    """
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if h.ndim != 1 or o.ndim != 1 or len(h) < 64 or not np.isfinite(h).all() or not np.isfinite(o).all():
        raise ValueError("Invalid series for Model69 transition statistics")
    if not len(o):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    edges = np.quantile(h, (0.25, 0.50, 0.75))
    hb = np.searchsorted(edges, h, side="right")
    ob = np.searchsorted(edges, o, side="right")
    historical = np.bincount(4 * hb[:-1] + hb[1:], minlength=16).reshape(4, 4)
    conditional = (historical + 1.0) / (historical.sum(axis=1, keepdims=True) + 4.0)
    prior = np.r_[hb[-1], ob[:-1]]
    code = 4 * prior + ob
    steps = len(o)
    cumulative = np.empty((steps + 1, 16), dtype=np.int32)
    cumulative[0] = 0
    for cell in range(16):
        cumulative[1:, cell] = np.cumsum(code == cell)
    end = np.arange(1, steps + 1)
    eye = np.eye(4, dtype=bool)
    up = np.triu(np.ones((4, 4), dtype=bool), 1)
    down = up.T
    blocks = []
    for window in WINDOWS:
        start = np.zeros_like(end) if window == 0 else np.maximum(0, end - window)
        counts = (cumulative[end] - cumulative[start]).reshape(-1, 4, 4).astype(np.float64)
        row_count = counts.sum(axis=2, keepdims=True)
        expected = row_count * conditional[None, :, :]
        glr = 2.0 * np.sum(counts * np.log(np.maximum(counts, 1.0) / np.maximum(expected, 1e-12)), axis=(1, 2))
        glr = np.maximum(glr, 0.0)
        residual = counts - expected
        norm = np.sqrt(np.maximum(end - start, 1))
        cell_z = residual / np.sqrt(np.maximum(expected, 1.0))
        block = np.column_stack((
            np.log1p(glr),
            (glr - 12.0) / np.sqrt(24.0),
            np.max(np.abs(cell_z), axis=(1, 2)),
            residual[:, eye].sum(axis=1) / norm,
            residual[:, up].sum(axis=1) / norm,
            residual[:, down].sum(axis=1) / norm,
        ))
        blocks.append(np.clip(block, -30.0, 30.0))
    result = np.ascontiguousarray(np.column_stack(blocks), dtype=np.float32)
    if result.shape != (steps, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Invalid Model69 transition features")
    return result
