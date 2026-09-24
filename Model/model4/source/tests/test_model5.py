"""Fast regression tests for the Model 5 selection and split protocol."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np

from models.model4.model4_pipeline import assign_series_splits
from models.model5 import config as editable_model5_config
from models.model5.config import FEATURE_SELECTION_CONFIG, make_model5_config
from models.model5.model5_pipeline import (
    MODEL5_FEATURE_SELECTION_VERSION,
    MODEL5_VERSION,
    Model5Config,
    detector_is_selected,
    detector_selection_bound,
    _screen_compatible_protocol,
)


PROJECT = Path(__file__).resolve().parents[1]


def _balanced_series_metadata() -> tuple[np.ndarray, np.ndarray]:
    lengths: list[int] = []
    tau: list[int] = []
    for _ in range(24):
        for length in (120, 240, 360):
            lengths.extend([length, length, length, length])
            tau.extend([-1, length // 6, length // 2, 5 * length // 6])
    return np.asarray(lengths), np.asarray(tau)


def test_model5_config_freezes_requested_protocol() -> None:
    config = make_model5_config(PROJECT)
    config.validate()
    FEATURE_SELECTION_CONFIG.validate()
    protocol = config.protocol()
    assert protocol["experiment_version"] == MODEL5_VERSION
    assert protocol["outer_folds"] == editable_model5_config.OUTER_FOLDS
    assert protocol["development_folds"] == editable_model5_config.CV_FOLDS
    assert protocol["development_from_outer_folds"] is True
    assert protocol["detector_selection_metric"] == "tsauc"
    assert protocol["detector_standard_error_multiplier"] == (
        editable_model5_config.DETECTOR_SE_MULTIPLIER
    )
    assert protocol["detector_roc_threshold"] == (
        editable_model5_config.DETECTOR_TSAUC_THRESHOLD
    )
    assert protocol["artifact_directory"] == config.artifact_directory
    assert protocol["artifact_directory"].startswith("models/model5/")
    assert config.feature_selection_workflow_version == MODEL5_FEATURE_SELECTION_VERSION
    feature_protocol = FEATURE_SELECTION_CONFIG.protocol(
        config.feature_selection_workflow_version
    )
    assert feature_protocol["feature_selection_version"] == (
        MODEL5_FEATURE_SELECTION_VERSION
    )
    assert feature_protocol["feature_counts"] == (
        editable_model5_config.FEATURE_COUNTS
    )
    assert feature_protocol["methods"] == (
        editable_model5_config.FEATURE_SELECTION_METHODS
    )


def test_model5_allows_configured_k_plus_one_outer_folds() -> None:
    Model5Config(
        project_root=PROJECT,
        outer_folds=5,
        development_folds=4,
    ).validate()
    try:
        Model5Config(
            project_root=PROJECT,
            outer_folds=5,
            development_folds=5,
        ).validate()
    except ValueError as error:
        assert "one outer holdout" in str(error)
    else:
        raise AssertionError("Model 5 accepted outer_folds != CV_FOLDS + 1")


def test_model5_half_se_rule_is_strict() -> None:
    assert detector_selection_bound(0.5101, 0.02, 0.5) > 0.5
    assert detector_is_selected(
        0.5101,
        standard_error=0.02,
        standard_error_multiplier=0.5,
    )
    assert detector_selection_bound(0.51, 0.02, 0.5) == 0.5
    assert not detector_is_selected(
        0.51,
        standard_error=0.02,
        standard_error_multiplier=0.5,
    )
    assert not detector_is_selected(
        0.5099,
        standard_error=0.02,
        standard_error_multiplier=0.5,
    )


def test_model5_screen_reuse_allows_only_postfit_cutoff_changes() -> None:
    first = {
        "artifact_directory": "models/model5/artifacts",
        "detector_roc_threshold": 0.5,
        "outer_seed": 7,
        "screen_num_boost_round": 100,
    }
    cutoff_only = {
        **first,
        "artifact_directory": "models/model5/runs/cutoff",
        "detector_roc_threshold": 0.55,
    }
    changed_split = {**cutoff_only, "outer_seed": 8}
    assert _screen_compatible_protocol(first) == _screen_compatible_protocol(
        cutoff_only
    )
    assert _screen_compatible_protocol(first) != _screen_compatible_protocol(
        changed_split
    )


def test_model5_uses_one_six_way_split_without_series_leakage() -> None:
    lengths, tau = _balanced_series_metadata()
    outer, inner = assign_series_splits(
        lengths,
        tau,
        outer_folds=6,
        holdout_fold=0,
        outer_seed=editable_model5_config.OUTER_SEED,
        development_folds=5,
        development_seed=editable_model5_config.CV_SEED,
        development_from_outer_folds=True,
    )
    assert set(np.unique(outer)) == set(range(6))
    assert np.all(inner[outer == 0] == -1)
    assert set(np.unique(inner[outer != 0])) == set(range(5))
    for outer_fold in range(1, 6):
        assert np.all(inner[outer == outer_fold] == outer_fold - 1)
    assert editable_model5_config.OUTER_SEED != 95_021


def test_model5_notebook_is_valid_and_code_cells_compile() -> None:
    notebook_path = PROJECT / "models" / "model5" / "model5.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 12
    all_source = "\n".join(cell.source for cell in notebook.cells)
    assert "flowchart TD" in all_source
    assert "make_model5_config" in all_source
    assert "CONFIG.detector_selection_metric == 'tsauc'" in all_source
    assert "search_feature_subsets" in all_source
    assert "evaluate_feature_selected_holdout" in all_source
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"model5.ipynb cell {index}", "exec")
