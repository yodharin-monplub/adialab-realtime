import numpy as np

from models.model37.report import bag_probability


def test_model37_bag_probability_contract():
    a = np.asarray([0.1, 0.4, 0.9], dtype=np.float32)
    b = np.asarray([0.3, 0.8, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(bag_probability(a, b), np.asarray([0.2, 0.6, 0.7], dtype=np.float32))
    np.testing.assert_array_equal(bag_probability(a, b, 0), a)
    np.testing.assert_array_equal(bag_probability(a, b, 1), b)

