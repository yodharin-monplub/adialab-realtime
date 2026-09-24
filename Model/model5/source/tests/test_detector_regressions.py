"""Regression tests for defects found by the all-detector audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_pipeline import (
    ALL_CANDIDATE_DETECTOR_BANKS,
    ALL_CORE_DETECTOR_BANKS,
    ALL_DETECTOR_BANKS,
    ARCHIVE_CANDIDATE_BANKS,
    ARCHIVE_CORE_BANKS,
    CANDIDATE_BANKS,
    CORE_BANKS,
    D3_COLUMNS_IN_D4,
    MODEL8_BANKS,
    NOVEL_BANKS,
    SPECIALIST_BANKS,
    CoreStreamDetector,
    _feature_signature,
    compute_features_offline,
    feature_names,
    d3,
    d4,
    d16,
)

import channels18
import detectors19 as d19
import detectors20 as d20


EXPECTED_PRODUCTION_SIGNATURES = {
    "core": "cde3ea9d9d3494cbf16d29c015afc4d53c598752a08aca03884f52a061287c54",
    "candidate": "78b1ada988001a974f1cdf151e5cf296e72ba959dc6903d30aa46e1e80eb8b25",
    "novel": "509c3e60c3ef7f326c6393e242c9145f51b0d02b9c3a6bf15de18963475c7465",
}


def _ar_series(seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(7.0, size=length)
    values = np.empty(length, dtype=np.float64)
    values[0] = noise[0]
    for index in range(1, length):
        values[index] = 0.57 * values[index - 1] + noise[index]
    return values


def test_channels18_matches_streaming_d3_innovations() -> None:
    values = _ar_series(1818, 1_180)
    history, online = values[:1_050], values[1_050:]
    offline = channels18.compute_channels_offline(history, online)
    detector = d3.StreamDetector3(history)
    rows = []
    for value in online:
        detector.step(value)
        position = (detector.t - 1) % detector._wmax
        rows.append(
            [
                np.clip(
                    detector._ring[aspect][position] / detector.hs.s[aspect],
                    -channels18.CLIP,
                    channels18.CLIP,
                )
                for aspect in d3.ASPECTS
            ]
        )
    streamed = np.asarray(rows)
    assert offline.shape == (online.size, len(d3.ASPECTS))
    assert np.max(np.abs(offline - streamed)) < 1e-12
    assert np.all(np.isfinite(channels18.compute_history_channels(history)))


def test_d16_uses_named_d3_columns_from_d4() -> None:
    values = _ar_series(1616, 1_160)
    history, online = values[:1_050], values[1_050:]
    base3 = d3.compute_features_offline(history, online)
    base4 = d4.compute_features_offline(history, online)

    # This is deliberately not a prefix: d4 inserts x99 columns at index 90.
    assert not np.array_equal(D3_COLUMNS_IN_D4, np.arange(d3.N_FEATURES))
    assert np.array_equal(base4[:, D3_COLUMNS_IN_D4], base3)

    expected = d16.compute_features_offline(history, online)[:, d3.N_FEATURES :]
    actual = compute_features_offline(history, online, ("d16",))
    detector = CoreStreamDetector(history, ("d16",))
    streamed = np.vstack([detector.step(value) for value in online])
    assert np.max(np.abs(actual - expected)) < 1e-12
    assert np.max(np.abs(streamed - expected)) < 1e-9


def test_restored_d19_and_d20_offline_streaming_parity() -> None:
    rng = np.random.default_rng(1920)
    values = _ar_series(1920, 1_160)
    history, online = values[:1_050], values[1_050:].copy()
    online[55:] = 1.4 * online[55:] + 0.3

    templates = rng.normal(size=(5, len(d3.ASPECTS)))
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    offline19 = d19.compute_features_offline(history, online, templates)
    stream19 = d19.StreamDetector19(history, templates)
    streamed19 = np.vstack([stream19.step(value) for value in online])
    assert np.max(np.abs(offline19 - streamed19)) < 1e-9

    model = d20._DummyEmission()
    offline20 = d20.compute_features_offline(history, online, model)
    stream20 = d20.StreamDetector20(history, model)
    streamed20 = np.vstack([stream20.step(value) for value in online])
    assert np.max(np.abs(offline20 - streamed20)) < 1e-9


def test_all_active_bank_groups_are_causal_finite_and_deterministic() -> None:
    values = _ar_series(4343, 1_125)
    history, online_a = values[:1_050], values[1_050:].copy()
    online_b = online_a.copy()
    online_b[36:] = 9.0 - 3.0 * online_b[36:]
    for banks in (
        CORE_BANKS,
        CANDIDATE_BANKS,
        SPECIALIST_BANKS,
        NOVEL_BANKS,
        MODEL8_BANKS,
    ):
        first = compute_features_offline(history, online_a, banks)
        repeated = compute_features_offline(history, online_a, banks)
        perturbed = compute_features_offline(history, online_b, banks)
        assert np.array_equal(first, repeated)
        assert np.array_equal(first[:36], perturbed[:36])
        assert np.all(np.isfinite(first))


def test_all_49_detector_sources_have_a_registered_bank() -> None:
    project = Path(__file__).resolve().parents[1]

    def detector_id(path: Path) -> str:
        suffix = path.stem.removeprefix("detectors")
        return f"d{int(suffix) if suffix else 1}"

    core_sources = {
        detector_id(path)
        for path in (project / "detector" / "core").glob("detectors*.py")
    }
    candidate_sources = {
        detector_id(path)
        for path in (project / "detector" / "candidate").glob("detectors*.py")
    }
    assert len(core_sources) == 28
    assert len(candidate_sources) == 21
    assert set(ALL_CORE_DETECTOR_BANKS) == core_sources
    assert set(ALL_CANDIDATE_DETECTOR_BANKS) == candidate_sources
    names = feature_names(ALL_DETECTOR_BANKS)
    assert len(names) == len(set(names))


def test_archived_model4_banks_are_causal_finite_and_streaming_equal() -> None:
    values = _ar_series(4344, 1_075)
    history, online = values[:1_050], values[1_050:]
    for banks in (ARCHIVE_CORE_BANKS, ARCHIVE_CANDIDATE_BANKS):
        offline = compute_features_offline(history, online, banks)
        detector = CoreStreamDetector(history, banks)
        streamed = np.vstack([detector.step(value) for value in online])
        assert offline.shape == (online.size, len(feature_names(banks)))
        assert np.all(np.isfinite(offline))
        assert np.max(np.abs(offline - streamed)) < 1e-8


def test_production_feature_contracts_are_unchanged() -> None:
    assert _feature_signature(CORE_BANKS) == EXPECTED_PRODUCTION_SIGNATURES["core"]
    assert (
        _feature_signature(CANDIDATE_BANKS)
        == EXPECTED_PRODUCTION_SIGNATURES["candidate"]
    )
    assert _feature_signature(NOVEL_BANKS) == EXPECTED_PRODUCTION_SIGNATURES["novel"]
