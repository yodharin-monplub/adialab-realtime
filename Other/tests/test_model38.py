import numpy as np

from models.model10.causal_transforms import StreamingCausalTransform
from models.model38.run_model38 import transform_spec


def test_model38_transform_is_frozen_short_memory_smoother():
    spec = transform_spec()
    assert spec.kind == "logit_ema_trend"
    assert spec.half_life == 4
    assert spec.correction_weight == -0.2
    stream = StreamingCausalTransform(spec)
    values = np.asarray([stream.step_probability(p, i) for i, p in enumerate([0.2, 0.8, 0.2])])
    assert np.isfinite(values).all()
    assert np.all((values >= 0) & (values <= 1))

