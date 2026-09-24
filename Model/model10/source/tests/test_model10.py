"""Fast synthetic tests for Model 10 causal score transformations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.model10.causal_transforms import (
    TransformSpec,
    apply_transform,
    apply_transform_grid,
    make_streaming_transform,
    safe_logit,
    score_to_probability,
    validate_causal_order,
)
from models.model10.config import MODEL10_VERSION, make_model10_config


PROJECT = Path(__file__).resolve().parents[1]


def _expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _synthetic_stream() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(
        [0.20, 0.25, 0.65, 0.48, 0.80, 0.75, 0.30, 0.31, 0.29, 0.70, 0.62],
        dtype=np.float64,
    )
    owner = np.repeat(np.arange(2, dtype=np.int64), [6, 5])
    time_online = np.concatenate(
        [np.arange(6, dtype=np.int64), np.arange(5, dtype=np.int64)]
    )
    return probability, time_online, owner


def test_model10_configuration_freezes_unique_transform_grid() -> None:
    config = make_model10_config(PROJECT)
    config.validate()
    specs = config.transform_specs()
    names = [spec.name for spec in specs]
    assert config.protocol()["experiment_version"] == MODEL10_VERSION
    assert specs[0].family == "raw"
    assert names[0] == "raw"
    assert len(specs) == 128
    assert len(names) == len(set(names))
    assert {spec.family for spec in specs} == {
        "raw",
        "logit_ema_trend",
        "rolling_max_premium",
        "decayed_peak_premium",
        "prior_ew_innovation",
    }
    assert config.artifact_directory.startswith("models/model10/runs/postprocess_")
    json.dumps(config.protocol(), sort_keys=True)


def test_model10_batch_and_streaming_transforms_match() -> None:
    probability, time_online, owner = _synthetic_stream()
    config = make_model10_config(PROJECT)
    for spec in config.transform_specs():
        offline = apply_transform(probability, time_online, owner, spec)
        streamed: list[float] = []
        for series in np.unique(owner):
            state = make_streaming_transform(spec)
            rows = np.flatnonzero(owner == series)
            streamed.extend(
                state.step(float(probability[row]), int(time_online[row]))
                for row in rows
            )
        np.testing.assert_allclose(
            offline,
            np.asarray(streamed, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
            err_msg=spec.name,
        )
        assert offline.dtype == np.float32
        assert np.all(np.isfinite(offline))


def test_model10_vectorized_grid_matches_every_reference_transform() -> None:
    probability, time_online, owner = _synthetic_stream()
    specs = make_model10_config(PROJECT).transform_specs()
    results = list(apply_transform_grid(probability, time_online, owner, specs))
    assert [spec.name for spec, _ in results] == [spec.name for spec in specs]
    for spec, vectorized in results:
        reference = apply_transform(probability, time_online, owner, spec)
        np.testing.assert_array_equal(vectorized, reference, err_msg=spec.name)


def test_model10_every_transform_is_strictly_causal() -> None:
    prefix = np.asarray([0.3, 0.4, 0.35, 0.7, 0.55], dtype=np.float64)
    future = np.asarray([0.99, 0.01, 0.95], dtype=np.float64)
    complete = np.concatenate((prefix, future))
    prefix_time = np.arange(prefix.size, dtype=np.int64)
    complete_time = np.arange(complete.size, dtype=np.int64)
    prefix_owner = np.zeros(prefix.size, dtype=np.int64)
    complete_owner = np.zeros(complete.size, dtype=np.int64)

    for spec in make_model10_config(PROJECT).transform_specs():
        before = apply_transform(prefix, prefix_time, prefix_owner, spec)
        after = apply_transform(complete, complete_time, complete_owner, spec)
        np.testing.assert_array_equal(before, after[: prefix.size], err_msg=spec.name)


def test_model10_state_resets_exactly_between_series() -> None:
    probability, time_online, owner = _synthetic_stream()
    specs = (
        TransformSpec(
            kind="logit_ema_trend", half_life=8.0, correction_weight=0.2
        ),
        TransformSpec(kind="rolling_max_premium", window=4, correction_weight=0.1),
        TransformSpec(
            kind="decayed_peak_premium",
            decay_factor=0.99,
            correction_weight=0.2,
        ),
        TransformSpec(
            kind="prior_ew_innovation",
            half_life=16.0,
            correction_weight=0.05,
            minimum_scale=0.25,
            innovation_clip=6.0,
        ),
    )
    for spec in specs:
        together = apply_transform(probability, time_online, owner, spec)
        pieces = []
        for series in (0, 1):
            rows = owner == series
            pieces.append(
                apply_transform(
                    probability[rows],
                    time_online[rows],
                    np.zeros(int(rows.sum()), dtype=np.int64),
                    spec,
                )
            )
        np.testing.assert_array_equal(together, np.concatenate(pieces))


def test_model10_corrections_preserve_raw_source_coefficient() -> None:
    # On a constant stream every history correction is zero. Therefore all
    # methods must return the unchanged raw logit, demonstrating that no
    # family silently rescales or removes the current source score.
    probability = np.full(9, 0.73, dtype=np.float64)
    time_online = np.arange(probability.size, dtype=np.int64)
    owner = np.zeros(probability.size, dtype=np.int64)
    expected = safe_logit(probability).astype(np.float32)
    specs = (
        TransformSpec(
            kind="logit_ema_trend", half_life=4.0, correction_weight=-0.35
        ),
        TransformSpec(kind="rolling_max_premium", window=8, correction_weight=0.2),
        TransformSpec(
            kind="decayed_peak_premium",
            decay_factor=0.99,
            correction_weight=0.2,
        ),
        TransformSpec(
            kind="prior_ew_innovation",
            half_life=64.0,
            correction_weight=0.2,
            minimum_scale=0.25,
            innovation_clip=6.0,
        ),
    )
    for spec in specs:
        np.testing.assert_array_equal(
            apply_transform(probability, time_online, owner, spec), expected
        )


def test_model10_rolling_max_uses_only_the_finite_window() -> None:
    probability = score_to_probability(np.asarray([0.0, 3.0, 0.0, 0.0, 0.0]))
    time_online = np.arange(5, dtype=np.int64)
    owner = np.zeros(5, dtype=np.int64)
    spec = TransformSpec(
        kind="rolling_max_premium", window=3, correction_weight=0.25
    )
    actual = apply_transform(probability, time_online, owner, spec)
    # The spike affects indices 1, 2, and 3, then expires before index 4.
    np.testing.assert_allclose(actual, [0.0, 3.0, 0.75, 0.75, 0.0], atol=1e-6)


def test_model10_probability_endpoints_remain_finite() -> None:
    probability = np.asarray([0.0, 1.0, 0.5, 0.0, 1.0], dtype=np.float64)
    time_online = np.arange(probability.size, dtype=np.int64)
    owner = np.zeros(probability.size, dtype=np.int64)
    for spec in make_model10_config(PROJECT).transform_specs():
        score = apply_transform(probability, time_online, owner, spec)
        converted = score_to_probability(score)
        assert np.all(np.isfinite(score))
        assert np.all(np.isfinite(converted))
        assert np.all((converted >= 0.0) & (converted <= 1.0))


def test_model10_rejects_bad_values_order_and_specs() -> None:
    probability, time_online, owner = _synthetic_stream()
    raw = TransformSpec(kind="raw")
    _expect_value_error(
        apply_transform,
        np.asarray([0.2, np.nan]),
        np.asarray([0, 1]),
        np.asarray([0, 0]),
        raw,
    )
    _expect_value_error(
        apply_transform,
        np.asarray([0.2, 1.2]),
        np.asarray([0, 1]),
        np.asarray([0, 0]),
        raw,
    )
    _expect_value_error(
        apply_transform,
        probability[:3],
        time_online[:3],
        np.asarray([0, 1, 0]),
        raw,
    )
    _expect_value_error(
        apply_transform,
        probability[:3],
        np.asarray([0, 2, 1]),
        np.asarray([0, 0, 0]),
        raw,
    )
    _expect_value_error(
        apply_transform,
        probability[:2],
        np.asarray([-1, 0]),
        np.asarray([0, 0]),
        raw,
    )
    _expect_value_error(
        validate_causal_order,
        probability[:2],
        np.asarray([0.0, 0.0]),
        time_online[:2],
    )
    malformed = (
        TransformSpec(kind="raw", correction_weight=0.1),
        TransformSpec(kind="logit_ema_trend", half_life=0.0, correction_weight=0.1),
        TransformSpec(kind="rolling_max_premium", window=1, correction_weight=0.1),
        TransformSpec(
            kind="decayed_peak_premium",
            decay_factor=0.0,
            correction_weight=0.1,
        ),
        TransformSpec(
            kind="prior_ew_innovation",
            half_life=16.0,
            correction_weight=0.1,
            minimum_scale=0.0,
            innovation_clip=6.0,
        ),
    )
    for spec in malformed:
        _expect_value_error(spec.validate)


__all__ = [name for name in globals() if name.startswith("test_")]
