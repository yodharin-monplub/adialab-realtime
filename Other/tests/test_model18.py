"""Small leakage, causality, reproducibility and checkpoint regression tests."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model18.augmentation import assemble, augmented_weights, real_fit_rows, select_pairs
from models.model18.run_model18 import commit_checkpoint, read_checkpoint, validate_predictions, CONFIG
from models.model18.report import simultaneous_bounds, choose_winner, false_alarms
from models.model11.features import compute_features, FEATURE_NAMES
from models.report_next_models import WeightedTSAUC
from tsauc import time_pair_weights, tsauc


class Model18Test(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(2)
        pool = {i: (rng.normal(i, 1, 80), rng.normal(i, 1, 120+i)) for i in range(18)}
        tau = np.full(20, 40)
        folds = np.r_[np.ones(18, dtype=int), 0, -1]
        return pool, tau, folds

    def test_pairs_deterministic_disjoint_unique_and_reuse(self):
        pool, tau, folds = self.fixture()
        pairs, stats = select_pairs(pool, tau, folds, 0, 91, count=100, reuse=2)
        self.assertEqual((pairs, stats), select_pairs(pool, tau, folds, 0, 91, count=100, reuse=2))
        self.assertEqual(len({p['a'] for p in pairs}), len(pairs))
        self.assertLessEqual(stats['max_b_reuse'], 2)
        self.assertEqual(stats['shortfall'], 100-len(pairs))
        for p in pairs:
            self.assertNotEqual(p['a'], p['b'])
            self.assertNotIn(folds[p['a']], (0, -1))
            self.assertNotIn(folds[p['b']], (0, -1))
            self.assertGreaterEqual(len(pool[p['b']][1])-p['tau_b'], p['remaining'])

    def test_forbidden_donors_rejected(self):
        for position in (18, 19):
            pool, tau, folds = self.fixture()
            pool[position] = pool[0]
            with self.assertRaisesRegex(ValueError, 'validation or reserved'):
                select_pairs(pool, tau, folds, 0, 2)

    def test_held_labels_cannot_affect_pairs(self):
        pool, tau, folds = self.fixture()
        expected = select_pairs(pool, tau, folds, 0, 12)
        tau[18:] = [-1, 3000]
        self.assertEqual(select_pairs(pool, tau, folds, 0, 12), expected)

    def test_float32_overflow_donors_are_rejected(self):
        pool, tau, folds = self.fixture()
        pool[0][0][:] = 1e100
        pool[0][1][:] = 1e100
        pairs, _ = select_pairs(pool, tau, folds, 0, 41)
        for pair in pairs:
            self.assertNotEqual(pair['a'], 0)
            self.assertNotEqual(pair['b'], 0)

    def test_nonfinite_and_short_donors(self):
        pool, tau, folds = self.fixture()
        pool[0][0][0] = np.nan
        pool[1][1][40] = np.inf
        tau[2] = 5
        pairs, _ = select_pairs(pool, tau, folds, 0, 5)
        for p in pairs:
            self.assertNotIn(p['a'], (0, 1, 2))
            self.assertNotIn(p['b'], (0, 1))

    def test_exact_assembly_alignment_labels_and_causality(self):
        pool, tau, folds = self.fixture()
        pairs, _ = select_pairs(pool, tau, folds, 0, 19)
        p = pairs[0]
        for aligned in (False, True):
            h, o, y, t = assemble(pool, p, aligned)
            self.assertEqual(len(o), len(pool[p['a']][1]))
            self.assertEqual(o.dtype, np.float32)
            np.testing.assert_array_equal(y, np.arange(len(o)) >= p['tau_a'])
            np.testing.assert_array_equal(o[:p['tau_a']], pool[p['a']][1][:p['tau_a']].astype(np.float32))
            tail = pool[p['b']][1][p['tau_b']:p['tau_b']+p['remaining']]
            np.testing.assert_array_equal(o[p['tau_a']:], (tail + (p['offset'] if aligned else 0)).astype(np.float32))
            am = np.median(np.r_[pool[p['a']][0], pool[p['a']][1][:p['tau_a']]][-128:])
            bm = np.median(np.r_[pool[p['b']][0], pool[p['b']][1][:p['tau_b']]][-128:])
            self.assertEqual(p['offset'], am-bm)
            features = compute_features(h, o)
            self.assertEqual(features.shape[1], 154)
            self.assertTrue(np.isfinite(features).all())
            np.testing.assert_array_equal(features[:60], compute_features(h, o[:60]))
            changed = o.copy()
            changed[60:] += 1000
            np.testing.assert_array_equal(features[:60], compute_features(h, changed)[:60])
        self.assertFalse(any(name in FEATURE_NAMES for name in ('tau', 'donor_id', 'length', 'offset')))

    def test_cell_mass_conservation_missing_and_zero_cells(self):
        ry = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
        rt = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        sy = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
        st = np.array([0, 0, 0, 3, 4])
        original = time_pair_weights(ry, rt)
        for share in (.1, .25):
            rw, sw, stats = augmented_weights(ry, rt, sy, st, share)
            before = np.bincount(2*rt+ry, weights=original, minlength=10)
            after = np.bincount(2*rt+ry, weights=rw, minlength=10)+np.bincount(2*st+sy, weights=sw, minlength=10)
            np.testing.assert_allclose(before, after, rtol=2e-7)
            np.testing.assert_array_equal(rw[rt == 1], original[rt == 1])
            self.assertEqual(sw[-1], 0)
            self.assertEqual(sw[-2], 0)
            self.assertLess(stats['actual_share'], share)

    def test_exact_control_row_selection_independent_of_augmentation(self):
        folds = np.repeat(np.arange(5), 100)
        for f in range(5):
            expected = np.sort(np.random.default_rng(20260906+f).choice(np.flatnonzero(folds != f), 90, replace=False))
            np.random.default_rng(20260918+f).random(1000)
            np.testing.assert_array_equal(real_fit_rows(folds, f, 90), expected)

    def test_checkpoint_resume_corruption_and_incomplete_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'fold.npz'
            meta = dict(arm='control')
            self.assertIsNone(read_checkpoint(path, meta))
            commit_checkpoint(path, dict(prediction=np.array([.2, .8]), held_rows=np.array([1, 2])), meta)
            saved = read_checkpoint(path, meta)
            validate_predictions(saved, np.array([1, 2]))
            with self.assertRaises(ValueError):
                validate_predictions(saved, np.array([2, 1]))
            with self.assertRaises(ValueError):
                read_checkpoint(path, dict(arm='raw_10'))
            with path.open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaises(ValueError):
                read_checkpoint(path, meta)

    def test_checkpoint_checks_model_artifact_and_uncommitted_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'fold.npz'
            artifact = Path(d)/'model.txt'
            artifact.write_text('model-v1')
            np.savez_compressed(path, prediction=np.array([.5]))
            self.assertIsNone(read_checkpoint(path, {}))
            commit_checkpoint(path, dict(prediction=np.array([.5])), {}, (artifact,))
            self.assertIsNotNone(read_checkpoint(path, {}))
            artifact.write_text('corrupted model')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                read_checkpoint(path, {})

    def test_false_alarm_rates_are_series_not_rows_and_exclude_reserved(self):
        layout = SimpleNamespace(inner_series_fold=np.array([0, 1, 0, -1]),
            tau_index=np.array([-1, -1, 1, -99]), series_position=np.array([0, 0, 1, 2, 2]))
        result = false_alarms(np.array([.1, .8, .4, .9, .9]), layout, [.5, .8])
        self.assertEqual(result['0.5']['series_count'], 2)
        self.assertEqual(result['0.5']['crossing_count'], 1)
        self.assertEqual(result['0.8']['crossing_rate'], .5)

    def test_bootstrap_resume_regenerates_identical_fold_draws(self):
        def draws(skip):
            rng = np.random.default_rng(180906)
            result = []
            for repeat in range(8):
                fold_draws = [rng.multinomial(n, np.full(n, 1/n)) for n in (7, 8, 6, 9, 5)]
                if repeat >= skip:
                    result.append(np.concatenate(fold_draws))
            return np.asarray(result)
        np.testing.assert_array_equal(draws(0)[3:], draws(3))
        lower, critical = simultaneous_bounds(np.zeros(4), np.zeros((1000, 4)))
        np.testing.assert_array_equal(lower, np.zeros(4))
        self.assertEqual(critical, 0)

    def test_whole_series_bootstrap_matches_literal_replication(self):
        y = np.array([0, 1, 0, 0, 1, 1], dtype=np.uint8)
        t = np.tile([0, 1], 3)
        s = np.repeat(np.arange(3), 2)
        prediction = np.array([.1, .5, .2, .5, .8, .9])
        count = np.array([2, 0, 3])
        rows = np.repeat(np.arange(6), count[s])
        scorer = WeightedTSAUC(y, prediction, t, s)
        self.assertAlmostEqual(scorer.score(count), tsauc(y[rows], prediction[rows], t[rows]), places=12)

    def test_centered_max_bounds_and_gate(self):
        observed = np.array([.01, .02, -.01, 0])
        noise = np.array([[.001, -.002, .003, 0], [-.003, .004, .002, -.001]])
        lower, critical = simultaneous_bounds(observed, observed+noise)
        self.assertAlmostEqual(critical, np.quantile([.003, .004], .95))
        np.testing.assert_allclose(lower, observed-critical)
        results = {name: dict(mean_gain=.002, fold_wins=4, simultaneous_lower_95=.001)
                   for name in ('raw_10', 'aligned_10', 'raw_25', 'aligned_25')}
        self.assertEqual(choose_winner(results, CONFIG), 'raw_10')
        for result in results.values():
            result['simultaneous_lower_95'] = 0
        self.assertIsNone(choose_winner(results, CONFIG))


if __name__ == '__main__':
    unittest.main()
