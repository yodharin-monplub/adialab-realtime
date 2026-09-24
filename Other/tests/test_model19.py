import unittest

import numpy as np
from scipy.special import logit

from models.model19.run_model19 import combine, passes, ARMS


class Model19Test(unittest.TestCase):
    def test_original_model16_formula_and_dtype(self):
        backbone = np.array([-.7, .1, 2.], dtype=np.float32)
        compact = np.array([.2, .5, .9], dtype=np.float32)
        expected = (.9*backbone+.1*logit(np.clip(compact, 1e-6, 1-1e-6))).astype(np.float32)
        np.testing.assert_array_equal(combine(backbone, compact), expected)
        self.assertEqual(combine(backbone, compact).dtype, np.float32)

    def test_only_compact_component_changes(self):
        backbone = np.linspace(-2, 2, 100).astype(np.float32)
        first, second = np.full(100, .4), np.full(100, .6)
        np.testing.assert_allclose(combine(backbone, second)-combine(backbone, first),
            .1*(logit(.6)-logit(.4)), atol=2e-7)
        second[50:] = .99
        np.testing.assert_array_equal(combine(backbone, second)[:50], combine(backbone, np.full(100, .6))[:50])

    def test_invalid_inputs(self):
        for b, p in (([1], [np.nan]), ([np.inf], [.5]), ([1], [1.1]), ([1], [.2, .3])):
            with self.assertRaises(ValueError):
                combine(b, p)

    def test_fixed_gates(self):
        result = dict(mean_gain=.0004, fold_wins=4, simultaneous_lower_95=.00001)
        self.assertTrue(passes(result))
        for key, value in [('mean_gain', .0002), ('fold_wins', 3), ('simultaneous_lower_95', 0)]:
            self.assertFalse(passes({**result, key: value}))
        self.assertEqual(ARMS, ('control', 'raw_10', 'aligned_10', 'raw_25', 'aligned_25'))


if __name__ == '__main__':
    unittest.main()
