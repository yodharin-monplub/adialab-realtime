"""Exact Time-Stratified AUC used by Structural Break: Real-Time.

The official scorer forms a cross-sectional ROC AUC independently at each
*online* step and averages those AUCs with ``n_positive * n_negative``
weights.  This module deliberately mirrors that implementation.  Keeping
the metric here (instead of using ordinary row-level AUC) prevents a common
and material validation error: observations from different online steps are
never compared by the competition.

The public scorer source is:
https://github.com/crunchdao/competitions/blob/master/competitions/
structural-break-real-time/scoring/scoring.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TSAUCResult:
    """A TS-AUC value plus useful accounting information."""

    score: float
    total_pair_weight: int
    scored_time_steps: int


def tsauc(
    target: Iterable[int],
    prediction: Iterable[float],
    time_online: Iterable[int],
) -> float:
    """Return the exact official TS-AUC value.

    Parameters
    ----------
    target:
        Binary labels.  A label is one from the break time onward.
    prediction:
        Finite real-valued scores.  Values need not be calibrated because
        ROC AUC depends only on ranking within a time stratum.
    time_online:
        Zero-based location inside the online segment.  This is *not* the
        original absolute time index in the parquet file.
    """

    return tsauc_details(target, prediction, time_online).score


def tsauc_details(
    target: Iterable[int],
    prediction: Iterable[float],
    time_online: Iterable[int],
) -> TSAUCResult:
    """Compute TS-AUC and return its pair count and stratum count."""

    y = np.asarray(target)
    score = np.asarray(prediction, dtype=np.float64)
    step = np.asarray(time_online)

    if y.ndim != 1 or score.ndim != 1 or step.ndim != 1:
        raise ValueError("target, prediction, and time_online must be 1-D")
    if not (y.size == score.size == step.size):
        raise ValueError("target, prediction, and time_online lengths differ")
    if y.size == 0:
        return TSAUCResult(0.5, 0, 0)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("target must contain only zero and one")
    if not np.all(np.isfinite(score)):
        raise ValueError("prediction contains NaN or infinity")

    # AUC is the normalized Mann--Whitney U statistic.  Sorting by
    # (time, score) once lets us sum U over all time strata without calling
    # sklearn hundreds of times.  Because the official pair weights are
    # n_positive*n_negative, its weighted numerator is exactly sum(U_t).
    order = np.lexsort((score, step))
    sorted_step = step[order]
    sorted_y = y[order]
    sorted_score = score[order]
    step_start = np.flatnonzero(
        np.r_[True, sorted_step[1:] != sorted_step[:-1]]
    )
    step_end = np.r_[step_start[1:], y.size]
    step_length = step_end - step_start
    positives = np.add.reduceat(sorted_y.astype(np.int64), step_start)
    negatives = step_length - positives
    pair_weights = positives * negatives

    # Local one-based ranks within each time group.  Positive observations
    # in an equal-score block receive the block's average rank, reproducing
    # sklearn's 0.5 credit for ties exactly.
    group_index = np.repeat(np.arange(step_start.size), step_length)
    local_rank = np.arange(y.size, dtype=np.float64) - step_start[group_index] + 1.0
    tie_start = np.flatnonzero(
        np.r_[
            True,
            (sorted_step[1:] != sorted_step[:-1])
            | (sorted_score[1:] != sorted_score[:-1]),
        ]
    )
    tie_end = np.r_[tie_start[1:], y.size]
    positive_in_tie = np.add.reduceat(
        sorted_y.astype(np.int64), tie_start
    )
    average_rank = 0.5 * (
        local_rank[tie_start] + local_rank[tie_end - 1]
    )
    positive_rank_sum = float(np.dot(positive_in_tie, average_rank))
    numerator = positive_rank_sum - float(
        np.sum(positives * (positives + 1) // 2)
    )
    denominator = int(pair_weights.sum())
    scored_steps = int(np.sum(pair_weights > 0))
    value = numerator / denominator if denominator else 0.5
    return TSAUCResult(float(value), int(denominator), int(scored_steps))


def fold_standard_error(scores: Iterable[float]) -> float:
    """Standard error of the mean across independent held-out folds."""

    values = np.asarray(list(scores), dtype=np.float64)
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def time_pair_weights(target: np.ndarray, time_online: np.ndarray) -> np.ndarray:
    """Pointwise training weights aligned with the TS-AUC pair geometry.

    At time ``t`` every positive participates in ``n_negative(t)`` pairs and
    every negative participates in ``n_positive(t)`` pairs.  These weights
    make a pointwise binary learner pay equal aggregate attention to both
    sides of every pair while preserving the official time-stratum weight.
    They are computed from a training fold only.
    """

    y = np.asarray(target, dtype=np.int8)
    step = np.asarray(time_online, dtype=np.int64)
    if y.shape != step.shape or y.ndim != 1:
        raise ValueError("target and time_online must be same-length 1-D arrays")
    if step.size == 0:
        return np.empty(0, dtype=np.float32)
    maximum = int(step.max())
    count = np.bincount(step, minlength=maximum + 1)
    positive = np.bincount(step, weights=y, minlength=maximum + 1)
    negative = count - positive
    weights = np.where(y == 1, negative[step], positive[step]).astype(np.float64)

    # The earliest strata may contain only negatives.  They carry no TS-AUC
    # pairs and therefore receive zero weight.  Normalize positive weights to
    # keep LightGBM's regularization scale stable across folds.
    nonzero = weights > 0
    if np.any(nonzero):
        weights[nonzero] /= weights[nonzero].mean()
    return weights.astype(np.float32)
