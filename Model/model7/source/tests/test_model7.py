"""Fast Model 7 contract tests; no learner is fitted."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np

from models.model7.config import MODEL7_VERSION, make_model7_config
from models.model7.model7_pipeline import (
    apply_absorbing_filter_grid,
    load_context,
    model_specs,
    query_group_layout,
    robust_score,
)


PROJECT = Path(__file__).resolve().parents[1]


def test_model7_configuration_declares_requested_experiment() -> None:
    config = make_model7_config(PROJECT)
    config.validate()
    assert config.protocol()["experiment_version"] == MODEL7_VERSION
    assert config.target_tsroc == 0.64
    assert len(config.repeat_seeds) == 3
    assert config.folds_per_repeat == 5
    assert config.feature_count == 256
    assert [spec.train_rows for spec in model_specs(config)] == [
        800_000,
        1_200_000,
        1_200_000,
    ]
    assert config.artifact_directory.startswith("models/model7/runs/")


def test_model7_repeats_exclude_revealed_holdout_series() -> None:
    context = load_context(make_model7_config(PROJECT))
    prohibited = context.source.split.holdout_series_mask
    assert np.all(context.repeat_series_fold[:, prohibited] == -1)
    allowed = context.source.split.development_series_mask
    for repeat in range(len(context.config.repeat_seeds)):
        assert set(context.repeat_series_fold[repeat, allowed]) == set(range(5))


def test_model7_query_weights_match_official_pair_geometry() -> None:
    target = np.asarray([0, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    time_online = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=np.int64)
    keep, weights = query_group_layout(target, time_online)
    assert keep.tolist() == [True] * 7 + [False, False]
    assert np.allclose(weights, [2.0 / 3.0, 4.0 / 3.0])


def test_model7_robust_score_penalizes_uncertainty() -> None:
    scores = [0.60, 0.62, 0.64]
    assert robust_score(scores, 0.5) < np.mean(scores)
    assert robust_score([0.62, 0.62, 0.62], 0.5) == 0.62


def test_model7_absorbing_filter_is_causal_and_finite() -> None:
    probability = np.asarray([0.2, 0.3, 0.8, 0.9, 0.4, 0.5], dtype=np.float64)
    time_online = np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64)
    series = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    calibration = {
        "edges": np.asarray([0, 2, 5], dtype=np.int64),
        "mean": np.asarray([[-1.0, -0.5], [1.0, 0.5]], dtype=np.float64),
        "scale": np.ones((2, 2), dtype=np.float64),
        "hazard": np.full(5, 0.02, dtype=np.float64),
    }
    first, settings = apply_absorbing_filter_grid(
        probability,
        time_online,
        series,
        calibration,
        (0.25, 0.50),
        (1.0,),
    )
    changed = probability.copy()
    changed[3] = 0.01
    second, _ = apply_absorbing_filter_grid(
        changed,
        time_online,
        series,
        calibration,
        (0.25, 0.50),
        (1.0,),
    )
    assert settings == [(0.25, 1.0), (0.5, 1.0)]
    assert np.all(np.isfinite(first))
    assert np.array_equal(first[:, :3], second[:, :3])


def test_model7_vectorized_filter_matches_serieswise_recurrence() -> None:
    probability = np.asarray([0.7, 0.2, 0.8, 0.3, 0.9, 0.4], dtype=np.float64)
    time_online = np.asarray([1, 0, 2, 0, 1, 2], dtype=np.int64)
    series = np.asarray([3, 9, 3, 3, 9, 9], dtype=np.int64)
    calibration = {
        "edges": np.asarray([0, 2, 5], dtype=np.int64),
        "mean": np.asarray([[-0.8, -0.4], [0.9, 0.6]], dtype=np.float64),
        "scale": np.asarray([[0.9, 1.1], [1.2, 0.8]], dtype=np.float64),
        "hazard": np.asarray([0.01, 0.03, 0.05, 0.07], dtype=np.float64),
    }
    actual, settings = apply_absorbing_filter_grid(
        probability, time_online, series, calibration, (0.2, 0.6), (0.5, 2.0)
    )

    expected = np.empty_like(actual)
    margin = np.clip(np.log(probability) - np.log1p(-probability), -8.0, 8.0)
    for setting, (evidence, hazard_scale) in enumerate(settings):
        state = 1e-5
        previous_series = -1
        for position in np.lexsort((time_online, series)):
            if series[position] != previous_series:
                state = 1e-5
                previous_series = int(series[position])
            online_time = int(time_online[position])
            time_bin = int(
                np.clip(
                    np.searchsorted(calibration["edges"], online_time, side="right")
                    - 1,
                    0,
                    1,
                )
            )
            value = margin[position]
            means = calibration["mean"][:, time_bin]
            scales = calibration["scale"][:, time_bin]
            ratio = (
                -0.5 * ((value - means[1]) / scales[1]) ** 2
                - np.log(scales[1])
                + 0.5 * ((value - means[0]) / scales[0]) ** 2
                + np.log(scales[0])
            )
            transition = np.clip(
                calibration["hazard"][online_time] * hazard_scale, 1e-7, 0.25
            )
            prior = np.clip(state + (1.0 - state) * transition, 1e-7, 1 - 1e-7)
            logit = np.log(prior) - np.log1p(-prior) + evidence * np.clip(
                ratio, -12.0, 12.0
            )
            state = 1.0 / (1.0 + np.exp(-logit))
            expected[setting, position] = state
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_model7_notebook_is_valid_and_cells_compile() -> None:
    notebook = nbformat.read(PROJECT / "models/model7/model7.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 5
    source = "\n".join(cell.source for cell in code_cells)
    assert "run_repeated_cv" in source
    assert "run_ensemble" in source
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"model7.ipynb cell {index}", "exec")
