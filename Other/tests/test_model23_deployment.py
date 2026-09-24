from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from models.model23.deployment.pairing import PairedXGBVoter
from models.model23.run_model23 import mix_probability


class Model23DeploymentTest(unittest.TestCase):
    def test_native_pair_matches_research_including_extremes(self):
        class FakeBooster:
            def __init__(self, p):
                self.p, self.inputs = p, []
            def inplace_predict(self, x, validate_features):
                self.inputs.append(x.copy())
                self.assertions = (x.dtype == np.float32 and x.flags.c_contiguous and not validate_features)
                return np.array([self.p], dtype=np.float32)
        row = np.arange(7, dtype=np.float64)
        columns = np.array([5, 1, 3])
        for a, b in [(0., .2), (1., .8), (.03, .87), (.5, .5)]:
            first, second = FakeBooster(a), FakeBooster(b)
            voter = PairedXGBVoter(first, second, columns)
            self.assertEqual(voter.predict(row), float(mix_probability([a], [b])[0]))
            self.assertTrue(first.assertions and second.assertions)
            np.testing.assert_array_equal(first.inputs[0], row[columns][None, :])
            np.testing.assert_array_equal(first.inputs[0], second.inputs[0])

    def test_pairing_before_logit_bag_is_not_flat_ten_voter_bag(self):
        a = np.array([.01, .2, .35, .6, .8], dtype=np.float32)
        b = np.array([.6, .7, .8, .9, .99], dtype=np.float32)
        def logit(p):
            return np.log(p)-np.log1p(-p)
        correct = np.mean(logit(mix_probability(a, b)))
        incorrect = np.mean(logit(np.r_[a, b]))
        self.assertGreater(abs(float(correct-incorrect)), .01)

    def test_entry_is_inference_only(self):
        calls = []
        module = SimpleNamespace(validate_resources=lambda path: calls.append(path))
        class NoData:
            def __iter__(self):
                raise AssertionError('Training data must not be read')
        entry = runpy.run_path(str(Path('models/model23/deployment/entry.py')))
        with patch.dict(sys.modules, solution23=module):
            entry['train'](NoData(), 'resources')
        self.assertEqual(calls, ['resources'])
        self.assertEqual(entry['INFER_PARALLELISM'], 12)


if __name__ == '__main__':
    unittest.main()
