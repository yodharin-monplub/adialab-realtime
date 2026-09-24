"""Small contract tests for the joint LightGBM experiment helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import joint_lightgbm_experiment as joint


def test_materialize_maps_global_columns_across_caches() -> None:
    data = SimpleNamespace(
        matrices=[
            np.arange(24, dtype=np.float32).reshape(6, 4),
            100 + np.arange(18, dtype=np.float32).reshape(6, 3),
        ],
        offsets=(0, 4),
        widths=(4, 3),
    )
    actual = joint._materialize(
        data, np.asarray([1, 4]), np.asarray([5, 0, 6, 3], dtype=np.int32)
    )
    expected = np.asarray(
        [[104, 4, 105, 7], [113, 16, 114, 19]], dtype=np.float32
    )
    assert np.array_equal(actual, expected)


def test_balanced_selection_reserves_novel_columns() -> None:
    data = SimpleNamespace(widths=(10, 12, 4), offsets=(0, 10, 22))
    order = np.arange(26, dtype=np.int32)
    selected = joint._balanced_columns(data, order, 20)
    assert selected.size == 20
    assert np.sum(selected >= 22) == 2
    assert len(set(selected.tolist())) == selected.size


def test_test_subfold_assignment_is_deterministic_and_balanced() -> None:
    series = np.repeat(np.arange(103), 3)
    first = joint._subfold_groups(series)
    second = joint._subfold_groups(series)
    assert np.array_equal(first, second)
    groups_by_series = first.reshape(-1, 3)[:, 0]
    counts = np.bincount(groups_by_series, minlength=5)
    assert counts.max() - counts.min() <= 1


def test_logit_blend_endpoints() -> None:
    first = np.asarray([0.1, 0.5, 0.9])
    second = np.asarray([0.8, 0.3, 0.2])
    assert np.allclose(joint._logistic(joint._safe_logit(first)), first)
    assert np.allclose(joint._logistic(joint._safe_logit(second)), second)
