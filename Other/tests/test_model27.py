import unittest
import numpy as np

from models.model27.augmentation import CONTEXT_NAMES, history_context, select_matched_pairs, assemble
from models.model27.run_model27 import candidate_probability


class Model27Test(unittest.TestCase):
    def test_fixed_float32_candidate_mixture(self):
        a=np.asarray([0,1,.1,.9],dtype=np.float32)
        b=np.asarray([1,0,.9,.1],dtype=np.float32)
        expected=np.float32(.8)*a+np.float32(.2)*b
        np.testing.assert_array_equal(candidate_probability(a,b),expected)

    def test_context_is_finite_scale_invariant_and_history_only(self):
        rng = np.random.default_rng(27)
        history = rng.standard_t(6, 500)
        first = history_context(history)
        second = history_context(3.5*history+8)
        self.assertEqual(first.shape, (len(CONTEXT_NAMES),))
        np.testing.assert_allclose(first[1:], second[1:], atol=1e-6, rtol=1e-6)
        self.assertTrue(np.isfinite(first).all())

    def test_pairs_exclude_held_stopping_reserved_and_make_paired_labels(self):
        rng = np.random.default_rng(28)
        pool = {}
        for p in range(24):
            history = rng.normal(size=160)
            online = rng.normal(size=96)
            online[48:] += (p%3)*.15
            pool[p] = (history, online)
        folds = np.asarray([-1, 0, 1, 2, 3, 4]*4)
        tau = np.full(24, 48)
        pairs, stats, _ = select_matched_pairs(pool, tau, folds, 0, 1, 91,
            count=5, reuse=2, candidates=8)
        self.assertGreater(len(pairs), 0)
        self.assertEqual(stats['shortfall'], 5-len(pairs))
        for pair in pairs:
            self.assertNotIn(folds[pair['a']], (-1, 0, 1))
            self.assertNotIn(folds[pair['b']], (-1, 0, 1))
            hp, op, yp, tp = assemble(pool, pair, True)
            hn, on, yn, tn = assemble(pool, pair, False)
            np.testing.assert_array_equal(hp, hn)
            np.testing.assert_array_equal(tp, tn)
            self.assertEqual(int(yp.sum()), pair['remaining'])
            self.assertEqual(int(yn.sum()), 0)
            self.assertEqual(len(op), pair['length'])
            self.assertEqual(len(on), pair['length'])


if __name__ == '__main__':
    unittest.main()
