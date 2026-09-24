"""Fast Model 8 contracts; no learner or feature cache is built."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np

from feature_pipeline import MODEL8_BANKS, feature_names
from models.model8.config import MODEL8_VERSION, make_model8_config
from models.model8.model8_pipeline import (
    _blend,
    build_frozen_outer_prediction,
    evaluate_outer_holdout,
    load_context,
    outer_voter_roles,
    role_rows,
    robust_score,
)


PROJECT = Path(__file__).resolve().parents[1]


def test_model8_configuration_freezes_new_detector_experiment() -> None:
    config = make_model8_config(PROJECT)
    config.validate()
    assert config.protocol()["experiment_version"] == MODEL8_VERSION
    assert config.detector_banks == MODEL8_BANKS
    assert len(feature_names(MODEL8_BANKS)) == 239
    assert config.folds == 5
    assert config.blend_weights[0] == 0.0
    assert config.blend_weights[-1] == 1.0


def test_model8_roles_exclude_revealed_holdout_and_each_other() -> None:
    context = load_context(make_model8_config(PROJECT), require_cache=False)
    owner = context.source.data.metadata["series_position"]
    prohibited = context.source.split.holdout_row_mask
    for fold in range(context.config.folds):
        fit, stop, score, _ = role_rows(context, fold)
        assert not np.any(prohibited[np.concatenate((fit, stop, score))])
        sets = [set(np.unique(owner[rows])) for rows in (fit, stop, score)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        outer_fit, outer_stop, outer_score = outer_voter_roles(context, fold)
        assert np.array_equal(np.sort(np.r_[outer_fit, outer_stop]), context.development_rows)
        assert np.array_equal(outer_score, context.source.split.holdout_rows)
        outer_sets = [set(np.unique(owner[rows])) for rows in (outer_fit, outer_stop, outer_score)]
        assert not (
            outer_sets[0] & outer_sets[1]
            or outer_sets[0] & outer_sets[2]
            or outer_sets[1] & outer_sets[2]
        )


def test_model8_blend_has_exact_endpoints_and_robust_penalty() -> None:
    baseline = np.asarray([0.2, 0.4, 0.7])
    candidate = np.asarray([0.8, 0.6, 0.3])
    assert np.allclose(_blend(baseline, candidate, 0.0), baseline)
    assert np.allclose(_blend(baseline, candidate, 1.0), candidate)
    assert robust_score([0.60, 0.62, 0.64], 0.5) < 0.62


def test_model8_notebook_is_valid_and_cells_compile() -> None:
    notebook = nbformat.read(PROJECT / "models/model8/model8.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 6
    source = "\n".join(cell.source for cell in code_cells)
    assert "run_detector_screen" in source
    assert "run_final_evaluation" in source
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"model8.ipynb cell {index}", "exec")


def test_model8_outer_prediction_and_scoring_are_separate() -> None:
    """The prediction stage cannot accidentally index outer target labels."""

    import inspect

    prediction_source = inspect.getsource(build_frozen_outer_prediction)
    evaluation_source = inspect.getsource(evaluate_outer_holdout)
    assert 'metadata["target"]' not in prediction_source
    assert 'metadata["target"][outer_rows]' in evaluation_source
    assert '"outer_holdout_accessed": True' in evaluation_source
