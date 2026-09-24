"""Causality, stream parity, metric, and pair-objective regression checks."""
from pathlib import Path
import sys
import json
import tempfile
import unittest

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model11.features import FEATURE_NAMES, StreamingFeatures, compute_features
from models.model11.pipeline import pairwise_derivatives, sample_pairs
from tsauc import tsauc
from models.model11.serving import infer
from models.model14.run_model14 import design, time_basis
from models.report_next_models import WeightedTSAUC


class CompactModelsTest(unittest.TestCase):
    def test_batch_matches_stream(self):
        rng = np.random.default_rng(31)
        for length in (3, 75, 900):
            for constant in (False, True):
                history = np.ones(length) if constant else np.cumsum(rng.normal(size=length))
                online = np.ones(310) if constant else rng.normal(size=310)
                state = StreamingFeatures(history)
                streaming = np.stack([state.update(v) for v in online])
                batch = compute_features(history, online)
                self.assertEqual(batch.shape, (310, len(FEATURE_NAMES)))
                np.testing.assert_allclose(batch, streaming, atol=2e-5, rtol=2e-5)
                self.assertTrue(np.isfinite(batch).all())

    def test_future_values_cannot_change_past_features(self):
        rng = np.random.default_rng(92)
        history = rng.normal(size=280)
        online = rng.normal(size=400)
        expected = compute_features(history, online)
        altered = online.copy()
        altered[170:] = rng.normal(100, 200, 230)
        np.testing.assert_array_equal(expected[:170], compute_features(history, altered)[:170])
        np.testing.assert_array_equal(expected[:170], compute_features(history, online[:170]))

    def test_reset_and_empty_online(self):
        history = np.arange(30, dtype=float)
        self.assertEqual(compute_features(history, []).shape, (0, len(FEATURE_NAMES)))
        first, second = StreamingFeatures(history), StreamingFeatures(history)
        first.update(3)
        np.testing.assert_array_equal(second.update(2), StreamingFeatures(history).update(2))

    def test_pair_sampler_excludes_other_times(self):
        target = np.array([0, 1, 1, 0, 0, 1, 0])
        step = np.array([0, 0, 0, 1, 1, 1, 2])
        positive, negative = sample_pairs(target, step, 1000, 19)
        self.assertTrue(np.all(target[positive] == 1))
        self.assertTrue(np.all(target[negative] == 0))
        np.testing.assert_array_equal(step[positive], step[negative])
        self.assertFalse(np.any(step[positive] == 2))
        p2, n2 = sample_pairs(target, step, 1000, 19)
        np.testing.assert_array_equal(positive, p2)
        np.testing.assert_array_equal(negative, n2)

    def test_pairwise_gradient_and_diagonal_hessian(self):
        scores = np.array([.2, -.5, 1., -.1])
        positive, negative = np.array([0, 0, 2]), np.array([1, 3, 1])
        gradient, hessian = pairwise_derivatives(scores, positive, negative)
        def loss(s):
            return np.logaddexp(0, s[negative]-s[positive]).sum() * len(s)/len(positive)
        epsilon = 1e-4
        for i in range(len(scores)):
            delta = np.eye(len(scores))[i]*epsilon
            self.assertAlmostEqual(gradient[i], (loss(scores+delta)-loss(scores-delta))/(2*epsilon), places=6)
            self.assertAlmostEqual(hessian[i], (loss(scores+delta)+loss(scores-delta)-2*loss(scores))/epsilon**2, places=5)
        self.assertAlmostEqual(gradient.sum(), 0)

    def test_metric_matches_official_time_loop_with_ties(self):
        rng = np.random.default_rng(97)
        y, time = rng.integers(0, 2, 250), np.tile(np.arange(10), 25)
        prediction = rng.integers(0, 5, 250).astype(float)
        numerator, denominator = 0., 0
        for t in np.unique(time):
            mask = time == t
            weight = int(y[mask].sum()) * int((1-y[mask]).sum())
            if weight:
                numerator += weight * roc_auc_score(y[mask], prediction[mask])
                denominator += weight
        self.assertAlmostEqual(tsauc(y, prediction, time), numerator/denominator, places=12)

    def test_time_interactions_use_only_current_step(self):
        basis = time_basis(np.array([0, 32, 128]))
        np.testing.assert_allclose(basis[:, 0], 1)
        np.testing.assert_allclose(basis[0], [1, .35, .35])
        self.assertTrue(np.all(np.diff(basis[:, 1:], axis=0) < 0))
        logits = np.array([[1., 2.], [3., 4.]])
        scale = np.array([2., 4.])
        matrix = design(logits, np.array([0, 32]), scale)
        np.testing.assert_allclose(matrix[:, :2], logits/scale)

    def test_weighted_auc_equals_literal_series_replication(self):
        series = np.repeat(np.arange(4), 3)
        step = np.tile(np.arange(3), 4)
        target = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        score = np.array([.2, .3, .4, .2, .4, .6, .2, .4, .5, .1, .3, .6])
        weights = np.array([2, 0, 3, 1])
        scorer = WeightedTSAUC(target, score, step, series)
        rows = np.repeat(np.arange(len(target)), weights[series])
        self.assertAlmostEqual(scorer.score(weights), tsauc(target[rows], score[rows], step[rows]), places=12)

    def test_serving_yields_before_consuming_next_observation(self):
        rng = np.random.default_rng(181)
        history, online = rng.normal(size=280), rng.normal(size=25)
        matrix = compute_features(history, online)
        synthetic = rng.normal(size=(100, len(FEATURE_NAMES))).astype(np.float32)
        model = lgb.train(dict(objective="binary", verbosity=-1, num_threads=1, min_data_in_leaf=3),
                          lgb.Dataset(synthetic, label=np.arange(100)%2), num_boost_round=5)
        expected = model.predict(matrix, num_threads=1)
        consumed = []
        def observations():
            for i, value in enumerate(online):
                consumed.append(i)
                yield value
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model.save_model(str(directory/"model.txt"))
            (directory/"manifest.json").write_text(json.dumps(dict(model=11, feature_names=FEATURE_NAMES)))
            output = infer([(history, observations())], str(directory), 11)
            self.assertIsNone(next(output))
            self.assertEqual(consumed, [])
            actual = []
            for i in range(len(online)):
                actual.append(next(output))
                self.assertEqual(len(consumed), i+1)
            with self.assertRaises(StopIteration):
                next(output)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
