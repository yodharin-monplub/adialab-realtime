"""Fast protocol tests for the model4 notebook helpers."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np

from models.model4 import config as editable_model4_config
from models.model4.config import FEATURE_SELECTION_CONFIG, make_model_config
from models.model4.feature_selection import (
    FeatureSelectionConfig,
    _aggregate_order,
    _rank_descending,
    choose_feature_winner,
)
from models.model4.model4_pipeline import (
    Model4Config,
    assign_series_splits,
    choose_primary_architecture,
    cluster_bootstrap_tsauc,
    development_fold_roles,
    detector_is_selected,
    nested_blend_selection,
)


PROJECT = Path(__file__).resolve().parents[1]


def _balanced_series_metadata() -> tuple[np.ndarray, np.ndarray]:
    lengths = []
    tau = []
    for repeat in range(20):
        for length in (100, 200, 300, 400, 500):
            lengths.append(length)
            tau.append(-1)
            for location in range(5):
                lengths.append(length)
                tau.append(int((location + 0.5) * length / 5.0))
    return np.asarray(lengths), np.asarray(tau)


def test_model4_editable_config_exposes_requested_search_grid() -> None:
    config = make_model_config(PROJECT)
    config.validate()
    FEATURE_SELECTION_CONFIG.validate()
    assert config.development_folds == editable_model4_config.CV_FOLDS
    assert config.detector_roc_threshold == (
        editable_model4_config.DETECTOR_ROC_THRESHOLD
    )
    assert FEATURE_SELECTION_CONFIG.feature_counts == (
        editable_model4_config.FEATURE_COUNTS
    )
    assert FEATURE_SELECTION_CONFIG.methods == (
        editable_model4_config.FEATURE_SELECTION_METHODS
    )


def test_model4_requires_train_stop_and_score_cv_roles() -> None:
    try:
        Model4Config(project_root=PROJECT, development_folds=2).validate()
    except ValueError as error:
        assert "at least three" in str(error)
    else:
        raise AssertionError("two-fold CV cannot provide train/stop/score roles")


def test_feature_selection_config_rejects_invalid_grids() -> None:
    invalid = (
        FeatureSelectionConfig(methods=("not-a-method",)),
        FeatureSelectionConfig(feature_counts=(64, 32)),
        FeatureSelectionConfig(feature_counts=(32, 32)),
    )
    for config in invalid:
        try:
            config.validate()
        except ValueError:
            continue
        raise AssertionError("invalid feature-selection grid was accepted")


def test_feature_rankings_and_borda_aggregation_are_deterministic() -> None:
    assert _rank_descending(np.asarray([1.0, 3.0, 3.0, np.nan])).tolist() == [1, 2, 0, 3]
    rankings = [
        {"gain": np.asarray([0, 1, 2, 3], dtype=np.int32)},
        {"gain": np.asarray([1, 0, 3, 2], dtype=np.int32)},
        {"gain": np.asarray([1, 2, 0, 3], dtype=np.int32)},
    ]
    assert _aggregate_order(rankings, "gain").tolist() == [1, 0, 2, 3]


def test_feature_winner_rule_uses_declared_tie_breaks() -> None:
    def candidate(method: str, count: int, mean: float, se: float) -> dict:
        return {
            "method": method,
            "feature_count": count,
            "tsauc": {"fold_mean": mean, "fold_standard_error": se},
        }

    candidates = [
        candidate("gain", 64, 0.61, 0.02),
        candidate("split", 32, 0.62, 0.03),
        candidate("consensus", 64, 0.62, 0.01),
        candidate("gain_corr", 32, 0.62, 0.01),
    ]
    winner, ordered = choose_feature_winner(
        candidates, ("gain", "split", "consensus", "gain_corr")
    )
    assert winner["method"] == "gain_corr"
    assert ordered[-1]["method"] == "gain"


def test_model4_series_split_is_reproducible_and_sealed() -> None:
    lengths, tau = _balanced_series_metadata()
    arguments = dict(
        outer_folds=5,
        holdout_fold=0,
        outer_seed=73051,
        development_folds=5,
        development_seed=84127,
    )
    outer_a, inner_a = assign_series_splits(lengths, tau, **arguments)
    outer_b, inner_b = assign_series_splits(lengths, tau, **arguments)
    assert np.array_equal(outer_a, outer_b)
    assert np.array_equal(inner_a, inner_b)
    assert np.all(inner_a[outer_a == 0] == -1)
    assert set(np.unique(inner_a[outer_a != 0])) == set(range(5))
    assert not np.any((outer_a == 0) & (inner_a >= 0))


def test_detector_selection_is_strictly_above_chance() -> None:
    assert detector_is_selected(0.5000001)
    assert not detector_is_selected(0.5)
    assert not detector_is_selected(0.4999999)
    assert not detector_is_selected(float("nan"))


def test_development_fit_has_disjoint_train_stop_and_score_roles() -> None:
    lengths, tau = _balanced_series_metadata()
    outer, inner = assign_series_splits(
        lengths,
        tau,
        outer_folds=5,
        holdout_fold=0,
        outer_seed=73051,
        development_folds=5,
        development_seed=84127,
    )
    # One synthetic row per series is enough to test the role contract.
    from models.model4.model4_pipeline import SplitAssignment

    split = SplitAssignment(
        outer_series_fold=outer,
        inner_series_fold=inner,
        row_inner_fold=inner.copy(),
        development_series_mask=outer != 0,
        holdout_series_mask=outer == 0,
        development_row_mask=outer != 0,
        holdout_row_mask=outer == 0,
        sampled_row_mask=np.ones(outer.size, dtype=bool),
    )
    for score_fold in range(5):
        train, stop, score, stop_fold = development_fold_roles(
            split, score_fold, 5
        )
        assert stop_fold == (score_fold + 1) % 5
        assert not np.any(train & stop)
        assert not np.any(train & score)
        assert not np.any(stop & score)
        assert np.array_equal(train | stop | score, split.development_row_mask)
        assert not np.any((train | stop | score) & split.holdout_row_mask)


def test_primary_architecture_rule_keeps_blend_on_tie() -> None:
    assert choose_primary_architecture(0.61, 0.62) == "joint"
    assert choose_primary_architecture(0.62, 0.61) == "blend"
    assert choose_primary_architecture(0.62, 0.62) == "blend"


def test_nested_blend_uses_only_other_fold_scores() -> None:
    rng = np.random.default_rng(44)
    fold = np.repeat(np.arange(5), 120)
    time_online = np.tile(np.repeat(np.arange(6), 20), 5)
    target = np.tile(np.tile(np.r_[np.zeros(10), np.ones(10)], 6), 5)
    core = np.clip(0.2 + 0.6 * target + rng.normal(0, 0.05, target.size), 0.01, 0.99)
    candidate = np.clip(rng.uniform(0.2, 0.8, target.size), 0.01, 0.99)
    result, prediction = nested_blend_selection(
        target,
        time_online,
        fold,
        core,
        candidate,
        grid_step=0.25,
    )
    assert prediction.shape == target.shape
    assert np.all(np.isfinite(prediction))
    assert len(result["core_weight_selected_without_held_fold"]) == 5
    assert 0.0 <= result["deployment_core_weight"] <= 1.0
    assert result["nested_tsauc"]["fold_mean"] > 0.9
    assert result["grid"][0] == 0.0
    assert result["grid"][-1] == 1.0


def test_nested_blend_grid_always_contains_both_endpoints() -> None:
    fold = np.repeat(np.arange(2), 12)
    time_online = np.tile(np.arange(6), 4)
    target = np.tile(np.r_[np.zeros(6), np.ones(6)], 2)
    core = 0.1 + 0.8 * target
    candidate = 1.0 - core
    result, _ = nested_blend_selection(
        target, time_online, fold, core, candidate, grid_step=0.3
    )
    assert result["grid"][0] == 0.0
    assert result["grid"][-1] == 1.0


def test_complete_series_bootstrap_is_deterministic() -> None:
    series = np.repeat(np.arange(20), 6)
    time_online = np.tile(np.arange(6), 20)
    target = (time_online >= (series % 5 + 1)).astype(np.uint8)
    prediction = 0.1 + 0.8 * target + 0.001 * series
    first = cluster_bootstrap_tsauc(
        target,
        prediction,
        time_online,
        series,
        replicates=20,
        seed=7,
    )
    second = cluster_bootstrap_tsauc(
        target,
        prediction,
        time_online,
        series,
        replicates=20,
        seed=7,
    )
    assert first == second
    assert first["series_clusters"] == 20
    assert first["standard_error"] >= 0.0


def test_model4_notebook_is_valid_and_code_cells_compile() -> None:
    notebook_path = PROJECT / "models" / "model4" / "model4.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    assert len(notebook.cells) >= 10
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 9
    notebook_source = "\n".join(cell.source for cell in code_cells)
    assert "make_model_config" in notebook_source
    assert "search_feature_subsets" in notebook_source
    assert "evaluate_feature_selected_holdout" in notebook_source
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"model4.ipynb cell {index}", "exec")
