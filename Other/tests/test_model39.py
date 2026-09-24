import numpy as np

from models.model38.run_model38 import transform_spec


def test_model39_reuses_model38_transform_without_retuning():
    spec = transform_spec()
    assert spec.kind == "logit_ema_trend"
    assert np.isclose(spec.half_life, 4)
    assert np.isclose(spec.correction_weight, -0.2)

