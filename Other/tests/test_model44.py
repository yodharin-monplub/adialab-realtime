import numpy as np

from models.model44.run_model44 import CONFIG


def test_model44_contract():
    assert CONFIG["fit_arms"] == ["aligned_25", "aligned_60"]
    assert CONFIG["pilot_folds"] == [0, 1]
    assert CONFIG["confirmation_folds"] == [2, 3, 4]
    assert CONFIG["blend_weights"] == sorted(CONFIG["blend_weights"])
    assert CONFIG["lightgbm"]["num_leaves"] == 31
    assert CONFIG["threads"] == 4


def test_blend_grid_is_conservative():
    weights = np.asarray(CONFIG["blend_weights"])
    assert np.all((weights > 0) & (weights <= 0.3))
