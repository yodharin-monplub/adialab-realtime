"""Causality and offline/online equality checks for the composite detector."""

from __future__ import annotations

import numpy as np

from feature_pipeline import (
    CANDIDATE_BANKS,
    MODEL8_BANKS,
    NOVEL_BANKS,
    SPECIALIST_BANKS,
    CoreStreamDetector,
    compute_features_offline,
    feature_names,
)


def _ar_series(seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovation = rng.standard_t(6.0, size=length)
    values = np.empty(length)
    values[0] = innovation[0]
    for index in range(1, length):
        values[index] = 0.62 * values[index - 1] + innovation[index]
    return values


def test_stream_equals_vectorized_offline() -> None:
    values = _ar_series(19, 1_450)
    historical = values[:1_100]
    online = values[1_100:].copy()
    online[173:] = 1.35 * online[173:] + 0.25
    offline = compute_features_offline(historical, online)
    detector = CoreStreamDetector(historical)
    streamed = np.vstack([detector.step(value) for value in online])
    assert offline.shape == (online.size, len(feature_names()))
    assert np.max(np.abs(offline - streamed)) < 1e-9


def test_future_values_cannot_change_earlier_features() -> None:
    values = _ar_series(41, 1_300)
    historical = values[:1_000]
    online_a = values[1_000:].copy()
    online_b = online_a.copy()
    online_b[120:] = 50.0 + 10.0 * online_b[120:]
    features_a = compute_features_offline(historical, online_a)
    features_b = compute_features_offline(historical, online_b)
    assert np.array_equal(features_a[:120], features_b[:120])


def test_features_are_finite_for_constant_history() -> None:
    historical = np.ones(1_000)
    online = np.r_[np.ones(20), np.full(20, 2.0)]
    features = compute_features_offline(historical, online)
    assert np.all(np.isfinite(features))


def test_candidate_stream_equals_vectorized_offline() -> None:
    values = _ar_series(57, 1_180)
    historical = values[:1_100]
    online = values[1_100:]
    offline = compute_features_offline(historical, online, CANDIDATE_BANKS)
    detector = CoreStreamDetector(historical, CANDIDATE_BANKS)
    streamed = np.vstack([detector.step(value) for value in online])
    assert offline.shape == (online.size, len(feature_names(CANDIDATE_BANKS)))
    assert np.max(np.abs(offline - streamed)) < 1e-9


def test_specialist_stream_equals_vectorized_offline() -> None:
    """Every optional specialist must preserve strict streaming parity."""

    values = _ar_series(73, 1_165)
    historical = values[:1_100]
    online = values[1_100:].copy()
    online[31:] *= 1.4
    offline = compute_features_offline(historical, online, SPECIALIST_BANKS)
    detector = CoreStreamDetector(historical, SPECIALIST_BANKS)
    streamed = np.vstack([detector.step(value) for value in online])
    assert offline.shape == (online.size, len(feature_names(SPECIALIST_BANKS)))
    assert np.max(np.abs(offline - streamed)) < 1e-9


def test_novel_winner_features_are_causal_and_streaming_equal() -> None:
    """The d60 history-null tests must never depend on a future online value."""

    values = _ar_series(109, 1_420)
    historical = values[:1_100]
    online_a = values[1_100:].copy()
    online_b = online_a.copy()
    online_b[140:] = 4.0 + 2.0 * online_b[140:]
    offline_a = compute_features_offline(historical, online_a, NOVEL_BANKS)
    offline_b = compute_features_offline(historical, online_b, NOVEL_BANKS)
    detector = CoreStreamDetector(historical, NOVEL_BANKS)
    streamed = np.vstack([detector.step(value) for value in online_a])
    assert offline_a.shape == (online_a.size, len(feature_names(NOVEL_BANKS)))
    assert np.array_equal(offline_a[:140], offline_b[:140])
    assert np.max(np.abs(offline_a - streamed)) < 1e-9
    assert np.all(np.isfinite(offline_a))


def test_model8_detector_banks_are_causal_finite_and_streaming_equal() -> None:
    """Every d61-d66 bank must preserve its standalone streaming contract."""

    values = _ar_series(118, 1_390)
    historical = values[:1_100]
    online_a = values[1_100:].copy()
    online_b = online_a.copy()
    online_b[145:] = -3.0 + 1.8 * online_b[145:]
    for bank in MODEL8_BANKS:
        banks = (bank,)
        offline = compute_features_offline(historical, online_a, banks)
        changed = compute_features_offline(historical, online_b, banks)
        detector = CoreStreamDetector(historical, banks)
        streamed = np.vstack([detector.step(value) for value in online_a])
        assert offline.shape == (online_a.size, len(feature_names(banks)))
        assert np.array_equal(offline[:145], changed[:145])
        assert np.max(np.abs(offline - streamed)) < 1e-8
        assert np.all(np.isfinite(offline))
