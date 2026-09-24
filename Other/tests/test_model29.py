import unittest
import numpy as np
from models.model29.run_model29 import CONFIG, meta_features


class Model29Tests(unittest.TestCase):
    def test_meta_features_are_finite_and_have_expected_width(self):
        base = np.asarray([.2, .8], dtype=np.float32)
        new = np.asarray([.3, .7], dtype=np.float32)
        context = np.arange(26, dtype=np.float32).reshape(2, 13)
        result = meta_features(base, new, context, np.asarray([0, 9]))
        self.assertEqual(result.shape, (2, 31))
        self.assertTrue(np.isfinite(result).all())

    def test_pilot_and_confirmation_are_disjoint(self):
        self.assertEqual(CONFIG["pilot_folds"], [0, 1])
        self.assertEqual(CONFIG["confirmation_folds"], [2, 3, 4])
        self.assertFalse(set(CONFIG["pilot_folds"]) & set(CONFIG["confirmation_folds"]))


if __name__ == "__main__":
    unittest.main()
