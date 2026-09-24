"""Fast contract tests for Model 6; no feature/model training is performed."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import nbformat
import numpy as np

import models.model6.run_model6 as model6_runner
from models.model6.config import MODEL6_VERSION, make_model6_config
from models.model6.model6_pipeline import (
    _deployment_transform,
    feature_contracts,
    hyperparameter_grid,
)
from models.model6.run_model6 import _compact_result


PROJECT = Path(__file__).resolve().parents[1]


def test_model6_configuration_covers_requested_search() -> None:
    config = make_model6_config(PROJECT)
    config.validate()
    assert config.protocol()["experiment_version"] == MODEL6_VERSION
    assert config.target_tsroc == 0.64
    assert config.source_detector_threshold == 0.55
    assert config.source_detector_se_multiplier == 0.5
    assert config.seed == 20_260_861
    assert config.probability_decay_factors[0] == 1.0
    assert config.probability_decay_factors[-1] == 0.5
    assert 0.99 in config.probability_decay_factors
    assert len(config.probability_decay_factors) > 3
    assert config.feature_counts == (32, 64, 128, 256, 512)
    assert set(config.ranking_methods) == {
        "gain",
        "split",
        "consensus",
        "gain_corr",
        "fisher",
    }
    assert {
        "lightgbm_gbdt",
        "lightgbm_dart",
        "lightgbm_goss",
        "lightgbm_rf",
        "xgboost",
        "catboost",
        "random_forest",
        "extra_trees",
        "linear_l2",
        "linear_l1",
        "linear_elasticnet",
    } == set(config.model_types)
    assert config.artifact_directory.startswith("models/model6/runs/")


def test_model6_feature_grid_has_all_once() -> None:
    config = make_model6_config(PROJECT)
    contracts = feature_contracts(config)
    assert len(contracts) == 5 * 5 + 1
    assert [row["key"] for row in contracts].count("all") == 1
    assert len({row["key"] for row in contracts}) == len(contracts)


def test_model6_every_learner_has_multiple_tuning_choices() -> None:
    config = make_model6_config(PROJECT)
    for model_type in config.model_types:
        grid = hyperparameter_grid(model_type)
        assert len(grid) >= 4
        assert len({row["name"] for row in grid}) == len(grid)


def test_model6_notebook_is_valid_and_code_cells_compile() -> None:
    notebook_path = PROJECT / "models" / "model6" / "model6.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 5
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "run_detector_stack" in source
    assert "run_feature_search" in source
    assert "run_hyperparameter_tuning" in source
    assert "freeze_deployment" in source
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"model6.ipynb cell {index}", "exec")


def test_model6_runner_declares_and_preflights_optional_learners() -> None:
    requirements = (PROJECT / "requirements.txt").read_text(encoding="utf-8")
    wrapper = (PROJECT / "models/model6/run_refresh_10000.sh").read_text(
        encoding="utf-8"
    )
    assert "xgboost-cpu" in requirements
    assert "catboost" in requirements
    assert '$PROJECT_ROOT/.venv/bin/python' in wrapper
    assert "import catboost, lightgbm, sklearn, xgboost" in wrapper


def test_model6_compact_cli_results_have_named_stages() -> None:
    status = _compact_result(
        {
            "run": "models/model6/runs/example",
            "development_series": 5,
            "sealed_holdout_series": 1,
        }
    )
    frozen = _compact_result(
        {
            "created_without_outer_holdout_scoring": True,
            "contract_hash": "abc",
            "development_nested_tsroc": {"fold_mean": 0.61},
        }
    )
    prediction = _compact_result(
        {
            "holdout_labels_read": False,
            "prediction_path": "prediction.npy",
            "frozen_contract_hash": "abc",
            "holdout_rows": 123,
        }
    )
    assert status["stage"] == "status"
    assert frozen["stage"] == "freeze_deployment"
    assert prediction["stage"] == "predict_holdout"


def test_model6_source_resume_still_builds_family_baselines() -> None:
    """A saved detector selection must not skip ranking prerequisites."""

    with TemporaryDirectory(prefix="model6-source-test-", dir="/tmp") as raw:
        artifact_path = Path(raw)
        selection = {
            "selection_hash": "selection-checkpoint",
            "selected_core_banks": ["d1"],
            "selected_candidate_banks": ["d2"],
        }
        (artifact_path / "selected_detectors.json").write_text(
            json.dumps(selection), encoding="utf-8"
        )
        source = SimpleNamespace(artifact_path=artifact_path)
        calls: list[tuple[object, object, object]] = []
        replacements = {
            "make_model5_config": lambda _root: source,
            "load_model5_data": lambda _config: "data",
            "prepare_split": lambda _data, _config: "split",
            "cross_validate_family_models": (
                lambda data, split, config: calls.append((data, split, config))
            ),
        }
        originals = {
            name: getattr(model6_runner, name) for name in replacements
        }
        try:
            for name, replacement in replacements.items():
                setattr(model6_runner, name, replacement)
            actual = model6_runner._prepare_model5_source()
        finally:
            for name, original in originals.items():
                setattr(model6_runner, name, original)

    assert actual == selection
    assert calls == [("data", "split", source)]


def test_model6_deployment_applies_probability_decay_contract() -> None:
    raw = np.asarray([0.8, 0.2, 0.1, 0.7, 0.4], dtype=np.float64)
    owner = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    report = {
        "postprocess": {
            "deployment_transform_spec": {
                "kind": "probability_multiplicative_decayed_max",
                "decay_factor": 0.5,
            }
        }
    }
    actual = _deployment_transform(raw, owner, report)
    assert np.allclose(actual, [0.8, 0.4, 0.2, 0.7, 0.4])
