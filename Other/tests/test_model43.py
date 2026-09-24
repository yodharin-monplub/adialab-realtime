import numpy as np
from models.model43.run_model43 import CONFIG


def test_model43_depth_contract():
    assert CONFIG["new_depth"] == 5
    np.testing.assert_allclose(sum(CONFIG["depth_weights"]), 1)

