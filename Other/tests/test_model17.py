"""Regression checks for fixed, causal blending and fold-excluded selection."""
import unittest
import numpy as np
from models.model17.study import RECIPES, blend, screening_gate, select_without_fold


class Model17Test(unittest.TestCase):
    def test_recipes_are_small_convex_blends(self):
        self.assertEqual(len(RECIPES), 8)
        self.assertEqual(RECIPES[0][1], (1., 0., 0.))
        for _, weights in RECIPES:
            self.assertAlmostEqual(sum(weights), 1.)
            self.assertGreaterEqual(weights[0], .85)
            self.assertTrue(all(w >= 0 for w in weights))

    def test_identity_and_weighted_result(self):
        margins = [np.array([1., 2., 3.], dtype=np.float32), np.array([-1., 4., 2.]), np.array([3., 1., 5.])]
        np.testing.assert_array_equal(blend(margins, (1, 0, 0)), margins[0])
        np.testing.assert_allclose(blend(margins, (.9, .05, .05)), .9*margins[0] + .05*margins[1] + .05*margins[2], atol=1e-6)

    def test_invalid_inputs_rejected(self):
        valid = [np.arange(3.)] * 3
        for weights in ((1, 1, 0), (1.1, -.1, 0), (float("nan"), 0, 0)):
            with self.assertRaises(ValueError): blend(valid, weights)
        with self.assertRaises(ValueError): blend([np.arange(3.), np.arange(2.), np.arange(3.)], (1, 0, 0))
        with self.assertRaises(ValueError): blend([np.array([np.nan])] * 3, (1, 0, 0))

    def test_future_predictions_do_not_change_prefix(self):
        rng = np.random.default_rng(17)
        margins = [rng.normal(size=100) for _ in range(3)]
        expected = blend(margins, (.85, .1, .05))
        for m in margins: m[50:] = 100
        np.testing.assert_array_equal(expected[:50], blend(margins, (.85, .1, .05))[:50])

    def test_held_fold_cannot_change_selection(self):
        scores = np.array([[.6, .6, .6], [.61, .62, .59], [.58, .58, .7]])
        for held in range(3):
            choice = select_without_fold(scores, held)
            changed = scores.copy(); changed[:, held] = [100, -100, 500]
            self.assertEqual(choice, select_without_fold(changed, held))
        self.assertEqual(select_without_fold(np.ones((3, 5)), 0), 0)

    def test_promotion_requires_all_gates(self):
        self.assertTrue(screening_gate(.0003, 4, .00001, .0001))
        for args in ((.0001, 5, .00001, .0001), (.0003, 3, .00001, .0001),
                     (.0003, 4, -.00001, .0001), (.0003, 4, .00001, 0)):
            self.assertFalse(screening_gate(*args))


if __name__ == "__main__":
    unittest.main()
