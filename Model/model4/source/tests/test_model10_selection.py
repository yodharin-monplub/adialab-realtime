"""Synthetic regression tests for Model 10's nested post-process selector."""

from __future__ import annotations

import numpy as np

from models.model10.selection import (
    NestedSelection,
    assemble_held_cell_predictions,
    expand_series_assignments,
    make_nested_series_subfolds,
    paired_standard_error,
    select_nested_candidates,
    summarize_foldwise_gain,
)


def _expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_nested_subfolds_are_series_complete_deterministic_and_exclude_holdout() -> None:
    # Each active original fold contains ten observations in each of five
    # horizon bins, enough for deterministic five-way stratification.
    original = np.repeat(np.asarray([-1, 0, 1, 2, 3, 4]), 50)
    lengths = np.tile(np.arange(50, 100, dtype=np.int64), 6)
    tau = np.full(original.size, -1, dtype=np.int64)
    # Deliberately invalid placeholders demonstrate that excluded holdout
    # metadata is never inspected by the calibration splitter.
    lengths[original == -1] = 0
    tau[original == -1] = -99

    first = make_nested_series_subfolds(original, lengths, tau, seed=913)
    repeated = make_nested_series_subfolds(original, lengths, tau, seed=913)
    changed_seed = make_nested_series_subfolds(original, lengths, tau, seed=914)
    np.testing.assert_array_equal(first, repeated)
    assert np.any(first[original >= 0] != changed_seed[original >= 0])
    assert np.all(first[original == -1] == -1)
    for fold in range(5):
        assert set(first[original == fold]) == set(range(5))

    row_owner = np.repeat(np.arange(original.size), 3)
    row_assignment = expand_series_assignments(row_owner, first)
    for series in range(original.size):
        assert np.unique(row_assignment[row_owner == series]).tolist() == [first[series]]


def test_nested_selection_uses_only_other_four_cells_with_raw_fallback() -> None:
    scores = np.full((3, 5, 5), 0.6, dtype=np.float64)
    scores[1] += 0.001
    scores[2] += 0.0001  # Below the predeclared minimum mean gain.
    # For held cell zero, candidate 1 is consistently good in training cells
    # 1..4.  Its bad held score cannot affect that choice.  For every other
    # held cell, the bad cell enters training and triggers raw fallback.
    scores[1, 0, 0] = 0.55

    selected = select_nested_candidates(scores)
    assert selected.selected_candidate[0].tolist() == [1, 0, 0, 0, 0]
    assert np.all(selected.selected_candidate[1:] == 1)
    assert abs(selected.training_mean_gain[0, 0] - 0.001) < 1e-12
    assert selected.training_standard_error[0, 0] < 1e-12

    modified_held_score = scores.copy()
    modified_held_score[1, 0, 0] = 0.99
    modified = select_nested_candidates(modified_held_score)
    # The choice for cell (0, 0) is invariant to that cell's own score.
    assert modified.selected_candidate[0, 0] == selected.selected_candidate[0, 0]
    assert (
        modified.training_robust_gain[0, 0]
        == selected.training_robust_gain[0, 0]
    )


def test_uncertainty_gate_rejects_unstable_candidate_even_above_mean_threshold() -> None:
    scores = np.full((2, 5, 5), 0.6, dtype=np.float64)
    # When cell zero is held, these four gains average +0.0005 but have enough
    # uncertainty that mean - 0.5*SE is negative.
    scores[1, 0, 1:] += np.asarray([0.004, 0.004, 0.004, -0.010])
    selected = select_nested_candidates(scores)
    assert selected.selected_candidate[0, 0] == 0
    assert selected.training_mean_gain[0, 0] == 0.0
    assert selected.training_robust_gain[0, 0] == 0.0


def test_held_prediction_assembly_uses_the_cell_specific_choice() -> None:
    chosen = np.tile(np.arange(5) % 3, (5, 1)).astype(np.int64)
    selection = NestedSelection(
        selected_candidate=chosen,
        training_mean_gain=np.zeros((5, 5)),
        training_standard_error=np.zeros((5, 5)),
        training_robust_gain=np.zeros((5, 5)),
        raw_index=0,
        uncertainty_penalty=0.5,
        minimum_mean_gain=0.00025,
        minimum_robust_gain=0.00010,
    )
    row_oof = np.repeat(np.arange(5), 5)
    row_nested = np.tile(np.arange(5), 5)
    predictions = np.vstack(
        [np.arange(25), 100 + np.arange(25), 200 + np.arange(25)]
    ).astype(np.float64)
    actual = assemble_held_cell_predictions(
        predictions, row_oof, row_nested, selection
    )
    expected_candidate = chosen[row_oof, row_nested]
    expected = predictions[expected_candidate, np.arange(25)]
    np.testing.assert_array_equal(actual, expected)

    excluded = row_oof.copy()
    excluded[0] = -1
    _expect_value_error(
        assemble_held_cell_predictions,
        predictions,
        excluded,
        row_nested,
        selection,
    )


def test_paired_fold_summary_reports_se_wins_and_promotion_gate() -> None:
    raw = np.asarray([0.61, 0.62, 0.63, 0.64, 0.60])
    gains = np.asarray([0.0010, 0.0005, 0.0004, 0.0003, -0.0001])
    summary = summarize_foldwise_gain(raw, raw + gains)
    np.testing.assert_allclose(summary.fold_gains, gains, rtol=0.0, atol=1e-15)
    assert abs(summary.paired_standard_error - paired_standard_error(gains)) < 1e-15
    assert summary.wins == 4
    assert summary.mean_gain >= 0.00025
    assert summary.robust_gain > 0.0
    assert summary.promotion_passed

    only_three_wins = np.asarray([0.0010, 0.0005, 0.0004, -0.0001, -0.0002])
    rejected = summarize_foldwise_gain(raw, raw + only_three_wins)
    assert rejected.mean_gain >= 0.00025
    assert rejected.robust_gain > 0.0
    assert rejected.wins == 3
    assert not rejected.promotion_passed


def test_selection_rejects_non_five_way_or_non_finite_inputs() -> None:
    _expect_value_error(select_nested_candidates, np.zeros((2, 5, 4)))
    malformed = np.zeros((2, 5, 5))
    malformed[1, 2, 3] = np.nan
    _expect_value_error(select_nested_candidates, malformed)
    _expect_value_error(paired_standard_error, [0.1])


__all__ = [name for name in globals() if name.startswith("test_")]
