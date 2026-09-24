"""Regression tests for the official metric replica."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from tsauc import fold_standard_error, time_pair_weights, tsauc


def test_tsauc_matches_literal_official_loop() -> None:
    rng = np.random.default_rng(123)
    lengths = np.asarray([5, 4, 5, 3, 2, 5])
    time = np.concatenate([np.arange(length) for length in lengths])
    target = rng.integers(0, 2, size=time.size)
    prediction = rng.normal(size=time.size)

    numerator = 0.0
    denominator = 0
    for step in np.unique(time):
        selected = time == step
        labels = target[selected]
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        if positives and negatives:
            weight = positives * negatives
            numerator += weight * roc_auc_score(labels, prediction[selected])
            denominator += weight
    expected = numerator / denominator
    assert tsauc(target, prediction, time) == expected


def test_time_only_score_is_random_within_each_scored_stratum() -> None:
    time = np.tile(np.arange(8), 20)
    target = np.repeat([0, 1], time.size // 2)
    assert tsauc(target, time.astype(float), time) == 0.5


def test_pair_weights_have_expected_class_totals() -> None:
    target = np.asarray([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    time = np.asarray([0, 0, 0, 1, 1, 1, 2, 2])
    weight = time_pair_weights(target, time)
    for step in np.unique(time):
        selected = time == step
        positives = selected & (target == 1)
        negatives = selected & (target == 0)
        if positives.any() and negatives.any():
            assert np.isclose(weight[positives].sum(), weight[negatives].sum())


def test_fold_standard_error() -> None:
    values = np.asarray([0.60, 0.61, 0.62, 0.63, 0.64])
    assert np.isclose(
        fold_standard_error(values), values.std(ddof=1) / np.sqrt(5)
    )

