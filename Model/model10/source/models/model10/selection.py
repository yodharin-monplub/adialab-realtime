"""Leakage-controlled selection for Model 10 score post-processing.

The five original development OOF folds are *not* a fresh outer holdout.  This
module subdivides the complete series in each such fold into five calibration
cells.  For a held cell, transform choice uses only the other four cells from
the same original OOF fold.  Series marked ``-1`` are excluded (for example a
previously sealed holdout) and no function here permits them in scoring.

All selection functions operate on ordinary NumPy arrays.  Artifact loading
and TSAUC calculation deliberately remain in the experiment pipeline so this
protocol can be tested without model files or competition data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


NESTED_SUBFOLDS = 5
DEFAULT_UNCERTAINTY_PENALTY = 0.5
DEFAULT_MINIMUM_MEAN_GAIN = 0.00025
DEFAULT_MINIMUM_ROBUST_GAIN = 0.00010
DEFAULT_MINIMUM_WINS = 4


def _as_integer_vector(value: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    """Return a one-dimensional integer view, rejecting silent truncation."""

    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return array.astype(np.int64, copy=False)


def make_nested_series_subfolds(
    original_oof_fold: Iterable[int] | np.ndarray,
    online_lengths: Iterable[int] | np.ndarray,
    tau_index: Iterable[int] | np.ndarray,
    *,
    seed: int,
    n_subfolds: int = NESTED_SUBFOLDS,
) -> np.ndarray:
    """Assign each development series to a five-way within-OOF subfold.

    ``original_oof_fold`` is a series-level array.  Non-negative values name
    the original OOF fold and ``-1`` means excluded data.  Each original fold
    is stratified independently, so calibration never borrows observations
    from another original fold.  The returned series-level assignments also
    retain ``-1`` for every excluded series.

    Stratification uses :func:`train_cv.make_series_folds`. Its inputs are the
    horizon and target-derived break presence/location, which is ordinary
    label-stratified split construction; no detector or prediction score enters
    the assignment. Importing it lazily keeps this utility cheap to import.
    """

    if n_subfolds != NESTED_SUBFOLDS:
        raise ValueError(f"Model 10 requires exactly {NESTED_SUBFOLDS} subfolds")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    original = _as_integer_vector(original_oof_fold, "original_oof_fold")
    lengths = _as_integer_vector(online_lengths, "online_lengths")
    tau = _as_integer_vector(tau_index, "tau_index")
    if original.shape != lengths.shape or original.shape != tau.shape:
        raise ValueError(
            "original_oof_fold, online_lengths, and tau_index must have equal shape"
        )
    if np.any(original < -1):
        raise ValueError("original_oof_fold may contain only -1 or non-negative folds")

    active = original >= 0
    if not np.any(active):
        raise ValueError("at least one non-excluded OOF series is required")
    # Do not inspect break metadata for excluded series.  In particular, a
    # previously sealed outer holdout can safely carry arbitrary placeholders.
    if np.any(lengths[active] <= 0):
        raise ValueError("active online_lengths must be positive")
    if np.any(tau[active] < -1) or np.any(tau[active] >= lengths[active]):
        raise ValueError("active tau_index must be -1 or inside its online horizon")

    from train_cv import make_series_folds

    nested = np.full(original.size, -1, dtype=np.int8)
    for fold in np.unique(original[active]):
        series = np.flatnonzero(original == fold)
        if series.size < n_subfolds:
            raise ValueError(
                f"OOF fold {int(fold)} has {series.size} series; "
                f"at least {n_subfolds} are required"
            )
        # A prime stride makes each original fold reproducible but gives it an
        # independent shuffle.  No target values or detector scores enter it.
        fold_seed = int(seed) + 104_729 * (int(fold) + 1)
        try:
            assignment = make_series_folds(
                lengths[series],
                tau[series],
                n_splits=n_subfolds,
                seed=fold_seed,
            )
        except ValueError as error:
            raise ValueError(
                f"cannot construct {n_subfolds} stratified subfolds inside "
                f"OOF fold {int(fold)}: {error}"
            ) from error
        assignment = _as_integer_vector(assignment, "nested assignment")
        if assignment.shape != series.shape:
            raise ValueError("make_series_folds returned an invalid assignment shape")
        if set(np.unique(assignment)) != set(range(n_subfolds)):
            raise ValueError("every original OOF fold must contain all five subfolds")
        nested[series] = assignment.astype(np.int8, copy=False)
    return nested


def expand_series_assignments(
    series_position: Iterable[int] | np.ndarray,
    series_assignment: Iterable[int] | np.ndarray,
) -> np.ndarray:
    """Expand a series-level fold assignment to rows without splitting series."""

    owner = _as_integer_vector(series_position, "series_position")
    assignment = _as_integer_vector(series_assignment, "series_assignment")
    if np.any(owner < 0) or (owner.size and int(owner.max()) >= assignment.size):
        raise ValueError("series_position refers to a missing series assignment")
    return assignment[owner]


def paired_standard_error(values: Iterable[float] | np.ndarray) -> float:
    """Return sample standard error for paired fold or cell differences."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("paired_standard_error requires at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError("paired differences must be finite")
    return float(array.std(ddof=1) / math.sqrt(array.size))


@dataclass(frozen=True)
class NestedSelection:
    """Transform chosen independently for every held calibration cell."""

    selected_candidate: np.ndarray
    training_mean_gain: np.ndarray
    training_standard_error: np.ndarray
    training_robust_gain: np.ndarray
    raw_index: int
    uncertainty_penalty: float
    minimum_mean_gain: float
    minimum_robust_gain: float


def select_nested_candidates(
    cell_scores: np.ndarray,
    *,
    raw_index: int = 0,
    uncertainty_penalty: float = DEFAULT_UNCERTAINTY_PENALTY,
    minimum_mean_gain: float = DEFAULT_MINIMUM_MEAN_GAIN,
    minimum_robust_gain: float = DEFAULT_MINIMUM_ROBUST_GAIN,
) -> NestedSelection:
    """Select one transform per held cell using its other four cells only.

    Parameters
    ----------
    cell_scores:
        Finite array shaped ``(candidate, original_oof_fold, 5)``.  Scores are
        computed separately on complete-series calibration cells.

    A non-raw candidate is eligible only when both its mean paired improvement
    is at least ``0.00025`` and ``mean - 0.5 * SE`` is at least ``0.00010``. The
    eligible candidate with the largest robust gain wins; ties retain the
    lowest candidate index.  If none qualifies, the raw endpoint is selected.
    """

    scores = np.asarray(cell_scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[2] != NESTED_SUBFOLDS:
        raise ValueError("cell_scores must have shape (candidate, fold, 5)")
    if scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError("cell_scores must contain candidates and OOF folds")
    if not np.all(np.isfinite(scores)):
        raise ValueError("cell_scores must be finite")
    if not isinstance(raw_index, (int, np.integer)) or not 0 <= raw_index < scores.shape[0]:
        raise ValueError("raw_index is outside cell_scores")
    if not math.isfinite(uncertainty_penalty) or uncertainty_penalty < 0.0:
        raise ValueError("uncertainty_penalty must be finite and non-negative")
    if not math.isfinite(minimum_mean_gain) or minimum_mean_gain < 0.0:
        raise ValueError("minimum_mean_gain must be finite and non-negative")
    if not math.isfinite(minimum_robust_gain) or minimum_robust_gain < 0.0:
        raise ValueError("minimum_robust_gain must be finite and non-negative")

    folds = scores.shape[1]
    chosen = np.full((folds, NESTED_SUBFOLDS), raw_index, dtype=np.int64)
    chosen_mean = np.zeros(chosen.shape, dtype=np.float64)
    chosen_se = np.zeros(chosen.shape, dtype=np.float64)
    chosen_robust = np.zeros(chosen.shape, dtype=np.float64)

    raw = scores[raw_index]
    for fold in range(folds):
        for held_cell in range(NESTED_SUBFOLDS):
            training_cells = np.arange(NESTED_SUBFOLDS) != held_cell
            gains = scores[:, fold, training_cells] - raw[fold, training_cells]
            means = gains.mean(axis=1)
            standard_errors = gains.std(axis=1, ddof=1) / math.sqrt(
                NESTED_SUBFOLDS - 1
            )
            robust = means - uncertainty_penalty * standard_errors
            eligible = (means >= minimum_mean_gain) & (
                robust >= minimum_robust_gain
            )
            eligible[raw_index] = False
            eligible_index = np.flatnonzero(eligible)
            if eligible_index.size:
                # np.argmax is stable: equal robust criteria favor the first
                # predeclared candidate and do not add a hidden tie-breaker.
                candidate = int(eligible_index[np.argmax(robust[eligible_index])])
                chosen[fold, held_cell] = candidate
                chosen_mean[fold, held_cell] = means[candidate]
                chosen_se[fold, held_cell] = standard_errors[candidate]
                chosen_robust[fold, held_cell] = robust[candidate]

    return NestedSelection(
        selected_candidate=chosen,
        training_mean_gain=chosen_mean,
        training_standard_error=chosen_se,
        training_robust_gain=chosen_robust,
        raw_index=int(raw_index),
        uncertainty_penalty=float(uncertainty_penalty),
        minimum_mean_gain=float(minimum_mean_gain),
        minimum_robust_gain=float(minimum_robust_gain),
    )


def assemble_held_cell_predictions(
    candidate_predictions: np.ndarray,
    row_oof_fold: Iterable[int] | np.ndarray,
    row_nested_subfold: Iterable[int] | np.ndarray,
    selection: NestedSelection,
) -> np.ndarray:
    """Assemble predictions using only each row's held-cell choice.

    Candidate predictions must be candidate-major with shape
    ``(candidate, row)``.  Rows marked ``-1`` are rejected rather than silently
    exposing a sealed holdout; callers must first filter to development OOF
    rows.
    """

    predictions = np.asarray(candidate_predictions)
    oof = _as_integer_vector(row_oof_fold, "row_oof_fold")
    nested = _as_integer_vector(row_nested_subfold, "row_nested_subfold")
    if predictions.ndim != 2 or predictions.shape[1] != oof.size:
        raise ValueError("candidate_predictions must have shape (candidate, row)")
    if nested.shape != oof.shape:
        raise ValueError("row_oof_fold and row_nested_subfold shapes differ")
    chosen = np.asarray(selection.selected_candidate)
    if chosen.ndim != 2 or chosen.shape[1] != NESTED_SUBFOLDS:
        raise ValueError("selection has an invalid selected_candidate shape")
    if np.any(oof < 0) or np.any(oof >= chosen.shape[0]):
        raise ValueError("rows must contain only scored development OOF folds")
    if np.any(nested < 0) or np.any(nested >= NESTED_SUBFOLDS):
        raise ValueError("rows must contain only the five scored nested subfolds")
    row_candidate = chosen[oof, nested]
    if np.any(row_candidate < 0) or np.any(row_candidate >= predictions.shape[0]):
        raise ValueError("selection refers to a missing candidate prediction")
    output = predictions[row_candidate, np.arange(oof.size)]
    if not np.all(np.isfinite(output)):
        raise ValueError("selected candidate predictions must be finite")
    return np.array(output, copy=True)


@dataclass(frozen=True)
class PairedGainSummary:
    """Foldwise uncertainty and the predeclared promotion decision."""

    raw_fold_scores: np.ndarray
    selected_fold_scores: np.ndarray
    fold_gains: np.ndarray
    raw_mean: float
    selected_mean: float
    mean_gain: float
    paired_standard_error: float
    robust_gain: float
    wins: int
    promotion_passed: bool


def summarize_foldwise_gain(
    raw_fold_scores: Iterable[float] | np.ndarray,
    selected_fold_scores: Iterable[float] | np.ndarray,
    *,
    uncertainty_penalty: float = DEFAULT_UNCERTAINTY_PENALTY,
    minimum_mean_gain: float = DEFAULT_MINIMUM_MEAN_GAIN,
    minimum_robust_gain: float = DEFAULT_MINIMUM_ROBUST_GAIN,
    minimum_wins: int = DEFAULT_MINIMUM_WINS,
) -> PairedGainSummary:
    """Summarize five paired OOF-fold changes and apply the promotion gate."""

    raw = np.asarray(raw_fold_scores, dtype=np.float64)
    selected = np.asarray(selected_fold_scores, dtype=np.float64)
    if raw.ndim != 1 or selected.shape != raw.shape or raw.size != 5:
        raise ValueError("raw and selected scores must be equal-length five-vectors")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(selected)):
        raise ValueError("fold scores must be finite")
    if not math.isfinite(uncertainty_penalty) or uncertainty_penalty < 0.0:
        raise ValueError("uncertainty_penalty must be finite and non-negative")
    if not math.isfinite(minimum_mean_gain) or minimum_mean_gain < 0.0:
        raise ValueError("minimum_mean_gain must be finite and non-negative")
    if not math.isfinite(minimum_robust_gain) or minimum_robust_gain < 0.0:
        raise ValueError("minimum_robust_gain must be finite and non-negative")
    if not isinstance(minimum_wins, (int, np.integer)) or not 0 <= minimum_wins <= 5:
        raise ValueError("minimum_wins must be an integer from zero through five")

    gains = selected - raw
    mean_gain = float(gains.mean())
    paired_se = paired_standard_error(gains)
    robust_gain = mean_gain - float(uncertainty_penalty) * paired_se
    wins = int(np.sum(gains > 0.0))
    promoted = bool(
        mean_gain >= minimum_mean_gain
        and robust_gain >= minimum_robust_gain
        and wins >= int(minimum_wins)
    )
    return PairedGainSummary(
        raw_fold_scores=raw.copy(),
        selected_fold_scores=selected.copy(),
        fold_gains=gains,
        raw_mean=float(raw.mean()),
        selected_mean=float(selected.mean()),
        mean_gain=mean_gain,
        paired_standard_error=paired_se,
        robust_gain=robust_gain,
        wins=wins,
        promotion_passed=promoted,
    )


__all__ = [
    "DEFAULT_MINIMUM_MEAN_GAIN",
    "DEFAULT_MINIMUM_ROBUST_GAIN",
    "DEFAULT_MINIMUM_WINS",
    "DEFAULT_UNCERTAINTY_PENALTY",
    "NESTED_SUBFOLDS",
    "NestedSelection",
    "PairedGainSummary",
    "assemble_held_cell_predictions",
    "expand_series_assignments",
    "make_nested_series_subfolds",
    "paired_standard_error",
    "select_nested_candidates",
    "summarize_foldwise_gain",
]
