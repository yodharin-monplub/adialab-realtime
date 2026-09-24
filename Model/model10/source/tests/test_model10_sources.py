"""Synthetic contract tests for Model 10 frozen prediction loaders."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from models.model10.sources import (
    DevelopmentLayout,
    load_development_layout,
    load_model6_incumbent_postprocess,
    load_model6_pre_global_postprocess,
    load_model9_pairwise_stack,
)
from tsauc import tsauc


def _layout() -> DevelopmentLayout:
    series = np.repeat(np.arange(4, dtype=np.int32), 2)
    target = np.asarray([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
    time = np.tile(np.arange(2, dtype=np.int32), 4)
    inner = np.asarray([0, 0, 1, 1], dtype=np.int8)
    return DevelopmentLayout(
        development_rows=np.arange(8, dtype=np.int64),
        target=target,
        time_online=time,
        series_position=series,
        row_fold=inner[series],
        inner_series_fold=inner,
        outer_series_fold=np.ones(4, dtype=np.int8),
        online_lengths=np.full(4, 2, dtype=np.int32),
        tau_index=np.asarray([-1, 0, 1, -1], dtype=np.int32),
        series_ids=np.arange(4, dtype=np.int64),
    )


def _scores(layout: DevelopmentLayout, prediction: np.ndarray) -> dict[str, object]:
    fold_scores = [
        tsauc(
            layout.target[layout.row_fold == fold],
            prediction[layout.row_fold == fold],
            layout.time_online[layout.row_fold == fold],
        )
        for fold in range(2)
    ]
    return {
        "fold_scores": fold_scores,
        "pooled_development_oof": tsauc(
            layout.target, prediction, layout.time_online
        ),
    }


def test_model10_reconstructs_model6_pre_postprocess_recipe() -> None:
    layout = _layout()
    first = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4], dtype=np.float32)
    second = np.asarray([0.2, 0.3, 0.7, 0.8, 0.4, 0.8, 0.7, 0.3], dtype=np.float32)
    expected = np.empty(8, dtype=np.float32)
    expected[layout.row_fold == 0] = first[layout.row_fold == 0]
    blend = 1.0 / (
        1.0
        + np.exp(
            -(
                0.75 * (np.log(first) - np.log1p(-first))
                + 0.25 * (np.log(second) - np.log1p(-second))
            )
        )
    )
    expected[layout.row_fold == 1] = blend[layout.row_fold == 1]

    with TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "model6"
        summary_path = run / "ensemble/global/summary.json"
        summary_path.parent.mkdir(parents=True)
        np.save(root / "first.npy", first)
        np.save(root / "second.npy", second)
        summary_path.write_text(
            json.dumps(
                {
                    "components": {"first": "first.npy", "second": "second.npy"},
                    "held_recipes": [
                        {
                            "held_fold": 0,
                            "recipe": {"start_component": "first", "steps": []},
                        },
                        {
                            "held_fold": 1,
                            "recipe": {
                                "start_component": "first",
                                "steps": [
                                    {
                                        "component": "second",
                                        "new_component_weight": 0.25,
                                    }
                                ],
                            },
                        },
                    ],
                    "nested_raw_tsroc": _scores(layout, expected),
                }
            ),
            encoding="utf-8",
        )
        source = load_model6_pre_global_postprocess(
            root, layout=layout, model6_directory="model6"
        )
    np.testing.assert_allclose(source.probability, expected, rtol=0.0, atol=1e-7)
    assert source.validation.maximum_absolute_error <= 1e-6


def test_model10_loads_model9_margin_as_rank_equivalent_probability() -> None:
    layout = _layout()
    margin = np.asarray([-2, -1, 2, 3, -0.5, 1, 0.5, -1], dtype=np.float32)
    probability = (1.0 / (1.0 + np.exp(-margin))).astype(np.float32)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "model9"
        run.mkdir(parents=True)
        np.save(run / "oof.npy", margin)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "pairwise_stack": _scores(layout, margin),
                    "oof_prediction_path": "model9/oof.npy",
                    "interpretation": "synthetic exploratory stack",
                }
            ),
            encoding="utf-8",
        )
        source = load_model9_pairwise_stack(
            root, layout=layout, model9_directory="model9"
        )
    np.testing.assert_allclose(source.probability, probability, rtol=1e-6, atol=1e-7)
    assert source.validation.maximum_absolute_error <= 1e-6


def test_model10_layout_materializes_only_filtered_development_labels() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "models").mkdir()
        (root / "train").mkdir()
        series = np.repeat(np.arange(4, dtype=np.int32), 2)
        np.savez(
            root / "models/metadata.npz",
            series_position=series,
            time_online=np.tile(np.arange(2, dtype=np.int16), 4),
            online_lengths=np.full(4, 2, dtype=np.int16),
            series_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        )
        np.savez(
            root / "models/splits.npz",
            inner_series_fold=np.asarray([-1, 0, 1, 0], dtype=np.int8),
            outer_series_fold=np.asarray([0, 1, 2, 1], dtype=np.int8),
        )
        pq.write_table(
            pa.table(
                {
                    "id": np.asarray([10, 11, 12, 13], dtype=np.int64),
                    "tau_index": np.asarray([0, -1, 1, 0], dtype=np.int64),
                }
            ),
            root / "train/index.parquet",
        )
        layout = load_development_layout(
            root,
            metadata_path="models/metadata.npz",
            split_path="models/splits.npz",
            target_index_path="train/index.parquet",
            folds=2,
        )
    assert layout.row_count == 6
    assert layout.tau_index.tolist() == [-99, -1, 1, 0]
    assert layout.target.tolist() == [0, 0, 0, 1, 1, 1]


def test_model10_loads_model6_incumbent_as_primary_baseline() -> None:
    layout = _layout()
    incumbent = np.asarray(
        [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4], dtype=np.float32
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        summary_path = root / "model6/ensemble/global/summary.json"
        summary_path.parent.mkdir(parents=True)
        np.save(root / "incumbent.npy", incumbent)
        summary_path.write_text(
            json.dumps(
                {
                    "ensemble_prediction_path": "incumbent.npy",
                    "postprocess": {
                        "deployment_transform": "probability_decayed_max_0.95",
                        "nested_tsroc": _scores(layout, incumbent),
                    },
                }
            ),
            encoding="utf-8",
        )
        source = load_model6_incumbent_postprocess(
            root, layout=layout, model6_directory="model6"
        )
    np.testing.assert_array_equal(source.probability, incumbent)
    assert source.name == "model6_incumbent_postprocess"
