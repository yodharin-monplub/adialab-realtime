import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from models.model20.run_model20 import reference_weights
from models.model22.run_model22 import (all_fitting_rows, additive_weights, row_digest,
    make_matrix, train_one, ARMS)
from models.model22.report import contrasts, passes


class Model22Test(unittest.TestCase):
    def test_every_eligible_real_row_retained(self):
        folds = np.repeat(np.arange(-1, 5), 150001)
        layout = SimpleNamespace(row_fold=folds)
        for fold in range(5):
            rows = all_fitting_rows(layout, fold)
            expected = np.flatnonzero(~np.isin(folds, [-1, fold, (fold+1) % 5]))
            np.testing.assert_array_equal(rows, expected)
            self.assertGreater(len(rows), 400000)
            self.assertEqual(len(rows), 3*150001)

    def test_addition_never_changes_real_weights(self):
        y = np.array([0, 0, 1, 0, 1, 1], dtype=np.uint8)
        t = np.array([0, 0, 0, 1, 1, 1])
        expected = reference_weights(y, t, y, t)
        real, synthetic, stats = additive_weights(y, t, y, t)
        np.testing.assert_array_equal(real, expected)
        np.testing.assert_array_equal(synthetic, expected)
        self.assertAlmostEqual(stats['combined_mass'], 2*stats['real_mass'])
        self.assertAlmostEqual(stats['synthetic_weight_share'], .5)
        larger_real, _, _ = additive_weights(y, t, np.tile(y, 8), np.tile(t, 8))
        np.testing.assert_array_equal(larger_real, expected)

    def test_unrepresented_cells_have_zero_synthetic_weight(self):
        y = np.array([0, 1, 0], dtype=np.uint8)
        t = np.array([0, 0, 1])
        real, synthetic, stats = additive_weights(y, t, np.array([1, 0, 1]), np.array([1, 7, 0]))
        self.assertEqual(synthetic[0], 0)
        self.assertEqual(synthetic[1], 0)
        self.assertEqual(synthetic[2], real[1])
        self.assertEqual(stats['synthetic_zero_weight_rows'], 2)

    def test_matrix_preserves_every_real_value_and_appends(self):
        real = np.arange(70, dtype=np.float32).reshape(10, 7)
        original = real.copy()
        rows = np.array([1, 2, 4, 8])
        columns = np.array([5, 2, 0])
        sx = np.arange(6, dtype=np.float32).reshape(2, 3)
        control = make_matrix(real, rows, columns)
        combined = make_matrix(real, rows, columns, sx)
        np.testing.assert_array_equal(control, real[np.ix_(rows, columns)])
        np.testing.assert_array_equal(combined[:len(rows)], control)
        np.testing.assert_array_equal(combined[len(rows):], sx)
        np.testing.assert_array_equal(real, original)

    def test_three_prespecified_contrasts_and_gate(self):
        np.testing.assert_allclose(contrasts([.63, .64, .642]), [.01, .012, .002])
        np.testing.assert_array_equal(contrasts([[1, 2, 4], [3, 2, 1]]), [[1, 3, 2], [-1, -2, -1]])
        good = dict(mean_gain=np.float64(.0005), fold_wins=4, simultaneous_lower_95=np.float64(.0001))
        self.assertIs(passes(good), True)
        json.dumps(dict(passes=passes(good)))
        self.assertIs(passes({**good, 'simultaneous_lower_95':0}), False)

    def test_training_inputs_match_and_resume_is_idempotent(self):
        recorded = []
        class RecordingModel:
            def __init__(self, **kwargs):
                pass
            def fit(self, x, y, sample_weight, **kwargs):
                recorded.append((x.copy(), y.copy(), sample_weight.copy()))
            def predict_proba(self, x):
                return np.full((len(x), 2), .5, dtype=np.float32)
            def save_model(self, path):
                Path(path).write_bytes(b'test-checkpoint')
        layout = SimpleNamespace(row_fold=np.repeat(np.arange(5), 6),
            target=((np.arange(30)+np.repeat(np.arange(5), 6)) % 2).astype(np.uint8),
            time_online=np.tile(np.arange(6), 5))
        rows = all_fitting_rows(layout, 0)
        plan = dict(fold=0, real_rows=len(rows), real_rows_sha256=row_digest(rows))
        recipe = dict(rounds=2, seed=10, hyperparameters={})
        protocol = dict(contract=dict(union_names=['a', 'b', 'c'], names=[['b', 'a']]*5, recipes=[recipe]*5))
        real = np.arange(90, dtype=np.float32).reshape(30, 3)
        synthetic = (np.zeros((3, 2), dtype=np.float32), np.array([0, 1, 0]), np.array([0, 1, 2]))
        with tempfile.TemporaryDirectory() as tmp, patch('models.model22.run_model22.xgb.XGBClassifier', RecordingModel):
            directory = Path(tmp)
            for arm in ARMS[1:]:
                train_one(directory, protocol, 'p', layout, real, plan, arm, synthetic)
            self.assertEqual(len(recorded), 2)
            for index in range(3):
                np.testing.assert_array_equal(recorded[0][index], recorded[1][index][:len(rows)])
            self.assertEqual(len(recorded[1][0]), len(rows)+3)
            self.assertGreater(recorded[0][2].sum(), 0)
            self.assertGreater(recorded[1][2].sum(), recorded[0][2].sum())
            for arm in ARMS[1:]:
                train_one(directory, protocol, 'p', layout, real, plan, arm, synthetic)
            self.assertEqual(len(recorded), 2)


if __name__ == '__main__':
    unittest.main()
