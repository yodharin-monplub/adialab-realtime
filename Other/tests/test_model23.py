import unittest
import numpy as np

from models.model23.run_model23 import ARMS, mix_probability


class Model23Test(unittest.TestCase):
    def test_fixed_equal_probability_mixture(self):
        self.assertEqual(ARMS, ('model20', 'mix_40', 'mix_60'))
        a = np.array([0, .1, .8, 1], dtype=np.float32)
        b = np.array([1, .3, .2, 0], dtype=np.float32)
        np.testing.assert_array_equal(mix_probability(a, a), a)
        np.testing.assert_array_equal(mix_probability(a, b), mix_probability(b, a))
        np.testing.assert_allclose(mix_probability(a, b), [.5, .2, .5, .5])
        self.assertEqual(mix_probability(a, b).dtype, np.float32)
        np.testing.assert_array_equal(a, [0, np.float32(.1), np.float32(.8), 1])

    def test_bad_vectors_rejected(self):
        for a, b in [([.5], [.2, .3]), ([[.5]], [[.2]]), ([np.nan], [.5]), ([-.1], [.5]), ([.5], [1.1])]:
            with self.assertRaises(ValueError):
                mix_probability(a, b)


if __name__ == '__main__':
    unittest.main()
