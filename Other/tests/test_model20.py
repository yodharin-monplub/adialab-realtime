import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from models.model20.features import selected_features, build_cache
from models.model20.run_model20 import sampled_mask, reference_weights, splice_weights, fitting_rows, synthetic_tasks
from models.model20.report import replace_stack_column, primitive_logit, contrasts
from models.model18.augmentation import select_pairs


class Model20Test(unittest.TestCase):
    def old_helper(self, name):
        path = Path('models/model6/source/train_cv.py')
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[name]

    def test_original_row_hash_and_reference_weights(self):
        rng = np.random.default_rng(2)
        sid = np.repeat(np.arange(40), 100)
        t = np.tile(np.arange(100), 40)
        np.testing.assert_array_equal(sampled_mask(sid, t), self.old_helper('deterministic_sample_mask')(sid, t, .2, 99173))
        y = rng.integers(0, 2, len(t), dtype=np.uint8)
        sample = rng.choice(len(t), 500, replace=False)
        np.testing.assert_array_equal(reference_weights(y[sample], t[sample], y, t),
            self.old_helper('pair_weights_from_reference')(y[sample], t[sample], y, t))

    def test_stop_and_validation_roles_excluded(self):
        series = np.repeat(np.arange(50), 100)
        layout = SimpleNamespace(series_ids=np.arange(50), series_position=series,
            row_fold=(series % 5), time_online=np.tile(np.arange(100), 50))
        for fold in range(5):
            selected, mask = fitting_rows(layout, fold, 100+fold)
            self.assertTrue(np.all(mask[selected]))
            self.assertFalse(np.isin(layout.row_fold[selected], [fold, (fold+1) % 5]).any())
            np.testing.assert_array_equal(selected, fitting_rows(layout, fold, 100+fold)[0])

    def test_full_reference_weight_mass_preserved(self):
        y = np.tile([0, 1], 30).astype(np.uint8)
        t = np.repeat(np.arange(30), 2)
        weights = reference_weights(y, t, np.r_[y, y[:20]], np.r_[t, t[:20]])
        sy, st = y[:40], t[:40]
        for share in (.1, .25):
            rw, sw, stats = splice_weights(weights, y, t, sy, st, share)
            before = np.bincount(2*t+y, weights=weights)
            after = np.bincount(2*t+y, weights=rw)+np.bincount(2*st+sy, weights=sw, minlength=len(before))
            np.testing.assert_allclose(after, before, rtol=2e-7)
            np.testing.assert_array_equal(rw[40:], weights[40:])
            self.assertLess(stats['actual_share'], share)

    def test_stack_identity_and_local_column_change(self):
        old = np.array([.2, .5, .8], dtype=np.float32)
        margin = np.array([-1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(replace_stack_column(margin, old, old, .3), margin)
        new = old.copy(); new[-1] = .9
        result = replace_stack_column(margin, old, new, .3)
        np.testing.assert_array_equal(result[:2], margin[:2])
        np.testing.assert_allclose(result, margin+.3*(primitive_logit(new)-primitive_logit(old)), atol=1e-7)
        np.testing.assert_array_equal(contrasts(np.arange(7.)), [1, 2, 2, 3])

    def test_donors_and_row_labels_match_across_modes(self):
        rng = np.random.default_rng(2)
        pool = {i: (rng.normal(size=100), rng.normal(size=120)) for i in range(9)}
        tau = np.full(9, 40)
        folds = np.full(9, 2)
        pairs, _ = select_pairs(pool, tau, folds, 0, 12, count=5)
        plan = dict(fold=0, pairs=pairs, selected=np.arange(0, 600, 2).tolist())
        layout = SimpleNamespace(inner_series_fold=folds)
        raw, y1, t1 = synthetic_tasks(pool, plan, layout, False)
        aligned, y2, t2 = synthetic_tasks(pool, plan, layout, True)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(t1, t2)
        for a, b in zip(raw, aligned):
            np.testing.assert_array_equal(a[2], b[2])
        folds[pairs[0]['b']] = 1
        with self.assertRaises(ValueError):
            synthetic_tasks(pool, plan, layout, False)

    def test_selected_backbone_features_are_causal(self):
        source = Path('models/model6/runs/tsroc0p64_909b8f5c6c/final_cv/core/xgboost__fisher_256__xgb_depth3')
        names = sorted(set(sum([np.load(source/f'fold_{f}.columns.npz')['names'].tolist() for f in range(5)], [])))
        rng = np.random.default_rng(7)
        # Deployed suffix-null detectors need a full clean reference, as in the data.
        history, online = rng.normal(size=1500), rng.normal(size=100)
        expected = selected_features(history, online, names)
        np.testing.assert_allclose(expected[:60], selected_features(history, online[:60], names), atol=1e-6, rtol=1e-6)
        online[60:] += 1000
        np.testing.assert_allclose(expected[:60], selected_features(history, online, names)[:60], atol=1e-6, rtol=1e-6)
        self.assertTrue(np.isfinite(expected).all())
        self.assertEqual(expected.shape, (100, 439))

    def test_cache_resume_and_corruption(self):
        rng = np.random.default_rng(3)
        tasks = [(rng.normal(size=100), rng.normal(size=20), None) for _ in range(2)]
        names = ['d1__S_up', 'd1__Q_up']
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            x = build_cache(directory, 'test', tasks, 40, names, 'protocol', workers=1)
            expected = x.copy()
            np.testing.assert_array_equal(build_cache(directory, 'test', tasks, 40, names, 'protocol', workers=1), expected)
            corrupt = np.load(directory/'test.npy', mmap_mode='r+')
            corrupt[0, 0] += 1
            corrupt.flush()
            with self.assertRaisesRegex(ValueError, 'Corrupted'):
                build_cache(directory, 'test', tasks, 40, names, 'protocol', workers=1)


if __name__ == '__main__':
    unittest.main()
