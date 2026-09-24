import numpy as np

from models.model11.pipeline import pairwise_derivatives
from models.model34.run_model34 import FEATURE_COLUMNS


def test_model34_feature_columns_exclude_incumbent_margin():
    assert len(FEATURE_COLUMNS) == 54
    assert 38 not in FEATURE_COLUMNS
    assert len(np.unique(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_pairwise_gradient_pushes_positive_above_negative():
    gradient, hessian = pairwise_derivatives(
        np.zeros(2), np.asarray([0]), np.asarray([1])
    )
    assert gradient[0] < 0 < gradient[1]
    assert np.all(hessian > 0)
