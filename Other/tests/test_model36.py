import numpy as np

from models.model36.arp_focus_features import (
    N_FEATURES, StreamDetector, compute_features_offline, feature_names,
)


def test_model36_feature_contract_and_stream_parity():
    rng = np.random.default_rng(36)
    history = rng.normal(size=600)
    online = rng.normal(size=90)
    online[45:] += .8
    offline = compute_features_offline(history, online)
    detector = StreamDetector(history)
    streamed = np.vstack([detector.step(value) for value in online])
    assert N_FEATURES == len(feature_names()) == 60
    np.testing.assert_allclose(offline, streamed, rtol=1e-10, atol=1e-10)
    assert np.isfinite(offline).all()


def test_model36_features_react_to_ar_mean_shift():
    rng = np.random.default_rng(3602)
    history = rng.normal(size=1000)
    unchanged = rng.normal(size=128)
    changed = unchanged.copy()
    changed[64:] += 2
    a = compute_features_offline(history, unchanged)
    b = compute_features_offline(history, changed)
    assert b[-1, 12] > a[-1, 12]
