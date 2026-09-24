"""Tests for grouped folds and deterministic sampling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from competition_data import TrainingData
import ensemble_training
from ensemble_cv import blend_predictions, safe_logit
from feature_pipeline import CORE_BANKS, feature_names
import solution
from train_cv import (
    HorizontalFeatureMatrix,
    deterministic_sample_mask,
    make_series_folds,
    matrix_rows,
)
from postprocess import (
    StreamingDecayedMax,
    StreamingProbabilityDecayedMax,
    causal_decayed_max,
    causal_probability_decayed_max,
)


def test_training_loader_checks_the_requested_label_file() -> None:
    """A loop-local offset must not shadow the requested ``train`` split name."""

    index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        names=["id", "time"],
    )
    features = pd.DataFrame(
        {
            "value": np.arange(6, dtype=np.float64),
            "period": [1, 2, 2, 1, 2, 2],
        },
        index=index,
    )
    break_index = pd.DataFrame(
        {"tau_index": [1, -1], "tau": [2, -1]},
        index=pd.Index([0, 1], name="id"),
    )
    # Series zero should be [0, 1], so these supplied labels are deliberately
    # wrong. TrainingData must read y_train.parquet and reject them.
    wrong_target = pd.DataFrame({"target": np.zeros(4, dtype=np.uint8)})
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        features.to_parquet(directory / "X_train.parquet")
        break_index.to_parquet(directory / "y_train_index.parquet")
        wrong_target.to_parquet(directory / "y_train.parquet")
        try:
            TrainingData(directory, split="train")
        except ValueError as error:
            assert "tau_index-derived labels disagree" in str(error)
        else:
            raise AssertionError("TrainingData failed to validate y_train.parquet")


def test_series_folds_are_balanced_and_complete() -> None:
    rng = np.random.default_rng(7)
    length = rng.integers(100, 1_000, size=1_000)
    has_break = np.arange(1_000) % 2 == 0
    tau = np.full(1_000, -1)
    tau[has_break] = (rng.random(has_break.sum()) * length[has_break]).astype(int)
    fold = make_series_folds(length, tau, n_splits=5, seed=11)
    assert set(fold) == set(range(5))
    assert np.all(np.bincount(fold) == 200)
    break_counts = np.bincount(fold[has_break], minlength=5)
    assert break_counts.max() - break_counts.min() <= 2


def test_sampling_is_reproducible_and_seeded() -> None:
    row_id = np.repeat(np.arange(100), 100)
    time = np.tile(np.arange(100), 100)
    first = deterministic_sample_mask(row_id, time, 0.2, 123)
    second = deterministic_sample_mask(row_id, time, 0.2, 123)
    different = deterministic_sample_mask(row_id, time, 0.2, 124)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert 0.18 < first.mean() < 0.22


def test_horizontal_matrix_reads_only_requested_global_columns_in_order() -> None:
    first = np.arange(30, dtype=np.float32).reshape(6, 5)
    second = 100 + np.arange(24, dtype=np.float32).reshape(6, 4)
    matrix = HorizontalFeatureMatrix([first, second])
    rows = np.asarray([5, 1, 3], dtype=np.int64)
    columns = np.asarray([8, 0, 6, 3], dtype=np.int32)
    expected = np.concatenate([first, second], axis=1)[np.ix_(rows, columns)]
    assert np.array_equal(matrix_rows(matrix, rows, columns), expected)


def test_vectorized_and_streaming_decayed_max_agree() -> None:
    probability = np.asarray([0.2, 0.8, 0.4, 0.3, 0.7, 0.1, 0.6])
    owner = np.asarray([0, 0, 0, 0, 1, 1, 1])
    expected = causal_decayed_max(probability, owner, penalty=0.15)
    actual = []
    for series in np.unique(owner):
        state = StreamingDecayedMax(0.15)
        actual.extend(state.step(value) for value in probability[owner == series])
    assert np.allclose(actual, expected)


def test_probability_decayed_max_matches_requested_recurrence() -> None:
    probability = np.asarray([0.8, 0.2, 0.1, 0.7, 0.4, 0.1])
    owner = np.asarray([0, 0, 0, 1, 1, 1])
    actual = causal_probability_decayed_max(
        probability, owner, decay_factor=0.5
    )
    # Each series resets, then applies state=max(raw, 0.5*previous_state).
    expected = np.asarray([0.8, 0.4, 0.2, 0.7, 0.4, 0.2])
    assert np.allclose(actual, expected)


def test_probability_decayed_max_batch_and_streaming_agree() -> None:
    probability = np.asarray([0.2, 0.8, 0.4, 0.3, 0.7, 0.1, 0.6])
    owner = np.asarray([0, 0, 0, 0, 1, 1, 1])
    for factor in (None, 1.0, 0.999, 0.99, 0.5):
        expected = causal_probability_decayed_max(probability, owner, factor)
        actual = []
        for series in np.unique(owner):
            state = StreamingProbabilityDecayedMax(factor)
            actual.extend(state.step(value) for value in probability[owner == series])
        assert np.allclose(actual, expected)


def test_probability_factor_one_is_a_per_series_running_maximum() -> None:
    probability = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3])
    owner = np.asarray([0, 0, 0, 1, 1])
    actual = causal_probability_decayed_max(probability, owner, 1.0)
    assert np.allclose(actual, [0.1, 0.9, 0.9, 0.8, 0.8])


def test_logit_ensemble_has_expected_endpoints() -> None:
    first = np.asarray([0.1, 0.4, 0.9])
    second = np.asarray([0.7, 0.2, 0.6])
    assert np.allclose(blend_predictions((first, second), (1, 0)), safe_logit(first))
    assert np.allclose(blend_predictions((first, second), (0, 1)), safe_logit(second))
    middle = blend_predictions((first, second), (1, 1))
    assert np.allclose(middle, 0.5 * (safe_logit(first) + safe_logit(second)))


def test_uploaded_ensemble_is_preserved_without_consuming_training_data() -> None:
    """Forced first-train must not replace an uploaded deployment ensemble."""

    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("pretrained deployment unexpectedly consumed data")

    with tempfile.TemporaryDirectory() as temporary_directory:
        model_directory = Path(temporary_directory)
        manifest_path = model_directory / "manifest.json"
        manifest = {
            "ensemble_method": "logit",
            "components": [{"directory": "core", "weight": 1.0}],
        }
        original_bytes = json.dumps(manifest).encode()
        manifest_path.write_bytes(original_bytes)

        validated_paths: list[str] = []
        original_loader = solution._load_ensemble
        solution._load_ensemble = lambda path: validated_paths.append(path)  # type: ignore[assignment]
        try:
            solution.train(MustNotIterate(), str(model_directory))
        finally:
            solution._load_ensemble = original_loader

        assert validated_paths == [str(model_directory)]
        assert manifest_path.read_bytes() == original_bytes


def test_selected_training_contract_matches_source_columns() -> None:
    """The frozen names must be an exact ordered projection of detector output."""

    available = feature_names(CORE_BANKS)
    columns = np.asarray([5, 1, 9], dtype=np.int32)
    selected = np.asarray([available[index] for index in columns])
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        np.savez(directory / "selection.npz", columns=columns, names=selected)
        specification = ensemble_training.ComponentSpec(
            "core",
            tuple(CORE_BANKS),
            boosting_rounds=1,
            selection_file="selection.npz",
            selection_method="test",
        )
        original_file = ensemble_training.__file__
        ensemble_training.__file__ = str(directory / "ensemble_training.py")
        try:
            names, actual_columns = ensemble_training._feature_contract(
                specification
            )
        finally:
            ensemble_training.__file__ = original_file

    assert names == selected.tolist()
    assert np.array_equal(actual_columns, columns)


def test_inference_loader_preserves_selected_feature_order() -> None:
    """Inference maps arbitrary selected names into the union detector row."""

    available = feature_names(("d4",))
    columns = [7, 2, 5]
    selected = [available[index] for index in columns]

    class FakeBooster:
        def __init__(self, model_file: str):
            assert Path(model_file).is_file()

        def num_feature(self) -> int:
            return len(selected)

    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        component = directory / "core"
        component.mkdir()
        (component / "final.txt").write_text("fake")
        (component / "manifest.json").write_text(
            json.dumps(
                {
                    "banks": ["d4"],
                    "feature_names": selected,
                    "feature_columns": columns,
                    "final_model": {"path": "final.txt"},
                }
            )
        )
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "ensemble_method": "logit",
                    "components": [{"directory": "core", "weight": 1.0}],
                }
            )
        )

        original_booster = solution.lgb.Booster
        solution.lgb.Booster = FakeBooster  # type: ignore[assignment]
        try:
            banks, components, method, penalty = solution._load_ensemble(
                str(directory)
            )
        finally:
            solution.lgb.Booster = original_booster

    assert banks == ("d4",)
    assert method == "logit"
    assert penalty is None
    assert len(components) == 1
    assert components[0].feature_names == tuple(selected)
    assert np.array_equal(components[0].columns, np.asarray(columns))
