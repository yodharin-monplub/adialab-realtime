import unittest
import numpy as np
from models.model26.deployment.pairing import PairedXGBVoter
from models.model26.run_model26 import candidate_probability


class Model26DeploymentTest(unittest.TestCase):
    def test_nested_float32_matches_research(self):
        class Booster:
            def __init__(self, p):
                self.p = p
            def inplace_predict(self, x, validate_features):
                assert x.dtype == np.float32 and x.flags.c_contiguous
                assert not validate_features
                np.testing.assert_array_equal(x, [[5, 1, 3]])
                return np.array([self.p], dtype=np.float32)
        rng = np.random.default_rng(26)
        cases = np.vstack(([0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0], rng.random((100, 4))))
        for values in cases:
            a, b, c, d = np.asarray(values, dtype=np.float32)
            base = np.float32(.5)*a + np.float32(.5)*b
            expected = candidate_probability(np.array([base]), np.array([c]), np.array([d]))[0]
            voter = PairedXGBVoter(*(Booster(p) for p in values), np.array([5, 1, 3]))
            self.assertEqual(voter.predict(np.arange(7)), float(expected))


if __name__ == '__main__':
    unittest.main()
