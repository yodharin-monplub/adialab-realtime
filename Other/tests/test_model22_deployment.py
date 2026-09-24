import copy
import unittest

import numpy as np

from models.model22.deployment.check_candidate import select_candidate, verify_real_identity


class Model22DeploymentTest(unittest.TestCase):
    def test_both_augmentation_comparisons_required(self):
        scores = np.array([[.62]*5, [.621]*5, [.622]*5])
        self.assertEqual(select_candidate(scores, [.0005, .001, .0005]), 'all_real_plus_synthetic')
        self.assertEqual(select_candidate(scores, [.0005, .001, -.0001]), 'all_real')
        self.assertIsNone(select_candidate(scores, [-.0001, -.0001, .0005]))

    def test_all_fold_win_and_effect_size_requirements(self):
        scores = np.array([[.62]*5, [.6201]*5, [.619, .619, .623, .623, .623]])
        self.assertIsNone(select_candidate(scores, [.0001, .0001, .0001]))
        for bad in (np.full((3, 5), np.nan), np.full((3, 5), 62.)):
            with self.assertRaises(ValueError):
                select_candidate(bad, [.001]*3)

    def test_identity_rejects_downweighting_and_changed_rows(self):
        control = dict(real_rows=2500000, real_rows_sha256='rows', real_weights_sha256='weights',
            synthetic_rows=0, weights=dict(real_mass=2500000., combined_mass=2500000., real_weights_unchanged=True))
        augmented = copy.deepcopy(control)
        augmented.update(synthetic_rows=400000)
        augmented['weights']['combined_mass'] = 2900000.
        verify_real_identity(control, augmented, 2500000)
        for key, value in [('real_rows_sha256', 'changed'), ('real_weights_sha256', 'changed'), ('real_rows', 400000)]:
            bad = copy.deepcopy(augmented)
            bad[key] = value
            with self.assertRaises(ValueError):
                verify_real_identity(control, bad, 2500000)
        bad = copy.deepcopy(augmented)
        bad['weights']['real_mass'] *= .75
        with self.assertRaises(ValueError):
            verify_real_identity(control, bad, 2500000)


if __name__ == '__main__':
    unittest.main()
