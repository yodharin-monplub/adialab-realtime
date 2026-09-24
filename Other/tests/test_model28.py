import unittest

import numpy as np

from models.model28.run_model28 import CONFIG, candidate_probability


class Model28Tests(unittest.TestCase):
    def test_candidate_is_exact_float32_80_percent_blend(self):
        base = np.asarray([.1, .4, .9], dtype=np.float32)
        new = np.asarray([.7, .2, .3], dtype=np.float32)
        expected = np.float32(.2) * base + np.float32(.8) * new
        actual = candidate_probability(base, new)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_pilot_is_disjoint_from_confirmation(self):
        self.assertEqual(CONFIG["pilot_fold"], 0)
        self.assertEqual(CONFIG["confirmation_folds"], [1, 2, 3, 4])
        self.assertNotIn(CONFIG["pilot_fold"], CONFIG["confirmation_folds"])


if __name__ == "__main__":
    unittest.main()
