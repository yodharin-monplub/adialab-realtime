import numpy as np

from models.model35.run_model35 import CONFIG


def test_model35_is_an_explicit_anti_ensemble():
    assert CONFIG["model34_residual_coefficient"] < 0
    assert CONFIG["model30_component_delta_coefficient"] < 0
    assert "adaptively" in CONFIG["selection_status"]
