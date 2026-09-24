import numpy as np

from models.model27.augmentation import CONTEXT_NAMES
from models.model33.run_model33 import feature_names, safe_logit


def test_model33_feature_contract_is_fixed_at_55_columns():
    names = feature_names([f"source_{index}" for index in range(34)])
    assert len(names) == 55
    assert names[-len(CONTEXT_NAMES):] == list(CONTEXT_NAMES)
    assert len(names) == len(set(names))


def test_model33_probability_logit_is_bounded_and_finite():
    values = safe_logit(np.asarray([0, 1e-12, .5, 1 - 1e-12, 1], dtype=np.float64))
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert values[0] == values[1]
    assert values[-1] == values[-2]
    assert values[2] == 0
