"""Small numerical helpers shared by the Model 8 detector banks.

The functions in this module are label-free.  Calibrations are always fitted
from rolling pseudo-suffixes of one series' clean history, then frozen before
the online segment begins.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


EPSILON = 1e-12
SCORE_CLIP = 12.0


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Return sums of every complete trailing window along axis zero."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim not in (1, 2):
        raise ValueError("rolling_sum accepts one- or two-dimensional arrays")
    if window < 1:
        raise ValueError("window must be positive")
    if matrix.shape[0] < window:
        return np.empty((0, *matrix.shape[1:]), dtype=np.float64)
    prefix_shape = (1, *matrix.shape[1:])
    cumulative = np.concatenate(
        (np.zeros(prefix_shape, dtype=np.float64), np.cumsum(matrix, axis=0)),
        axis=0,
    )
    return cumulative[window:] - cumulative[:-window]


def align_complete(values: np.ndarray, window: int, length: int) -> np.ndarray:
    """Align complete-window values to their right endpoint; prefix with zero."""

    raw = np.asarray(values, dtype=np.float64)
    output = np.zeros((length, *raw.shape[1:]), dtype=np.float64)
    if raw.shape[0]:
        output[window - 1 :] = raw
    return output


def fit_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columnwise finite mean/SD with deterministic degenerate fallbacks."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    finite = np.where(np.isfinite(matrix), matrix, np.nan)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(finite, axis=0)
        scale = np.nanstd(finite, axis=0)
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    scale = np.where(scale > 1e-7, scale, 1.0)
    return center, scale


def standardize(
    values: np.ndarray,
    center: np.ndarray | float,
    scale: np.ndarray | float,
) -> np.ndarray:
    """Apply a frozen history calibration and clip numerical extremes."""

    score = (np.asarray(values, dtype=np.float64) - center) / scale
    return np.clip(
        np.nan_to_num(score, nan=0.0, posinf=SCORE_CLIP, neginf=-SCORE_CLIP),
        -SCORE_CLIP,
        SCORE_CLIP,
    )


def frozen_midrank(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values to deterministic midranks against a frozen reference sample."""

    reference = np.asarray(sorted_reference, dtype=np.float64)
    query = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(reference, query, side="left")
    right = np.searchsorted(reference, query, side="right")
    return (left + right + 1.0) / (2.0 * (reference.size + 1.0))


def entropy(probability: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy with the convention ``0 log 0 = 0``."""

    p = np.asarray(probability, dtype=np.float64)
    term = np.zeros_like(p)
    positive = p > 0
    term[positive] = p[positive] * np.log(p[positive])
    return -np.sum(term, axis=axis)


def aggregate_maxima(
    columns_by_metric: dict[str, Sequence[np.ndarray]],
    metric_order: Sequence[str],
) -> list[np.ndarray]:
    """Return max-absolute and signed-at-max columns for each metric."""

    output: list[np.ndarray] = []
    for metric in metric_order:
        matrix = np.column_stack(columns_by_metric[metric])
        index = np.argmax(np.abs(matrix), axis=1)
        signed = matrix[np.arange(matrix.shape[0]), index]
        output.extend((np.abs(signed), signed))
    return output


def last_maxima(
    values_by_metric: dict[str, Sequence[float]], metric_order: Sequence[str]
) -> list[float]:
    """Scalar counterpart of :func:`aggregate_maxima` for streaming rows."""

    output: list[float] = []
    for metric in metric_order:
        values = np.asarray(values_by_metric[metric], dtype=np.float64)
        index = int(np.argmax(np.abs(values)))
        output.extend((float(abs(values[index])), float(values[index])))
    return output
