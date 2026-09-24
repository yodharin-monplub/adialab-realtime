from types import SimpleNamespace
import unittest
from models.model23.deployment.check_cloud import score_record

class CloudScoreTest(unittest.TestCase):
    def test_raw_api_metric_metadata(self):
        score=SimpleNamespace(_attrs={'value':62.37894019876368,
            'metric':{'name':'ts-auc','multiplier':100.0}},metric=None)
        result=score_record(score)
        self.assertAlmostEqual(result['unscaled_value'],.6237894019876368)
        self.assertAlmostEqual(result['gain_vs_model20'],.0003412965595006)
    def test_percentage_scale(self):
        score=SimpleNamespace(_attrs={'value':62.34481054281362},
            metric=SimpleNamespace(_attrs={'name':'ts-auc','multiplier':100}))
        result=score_record(score)
        self.assertAlmostEqual(result['unscaled_value'],.6234481054281362)
        self.assertAlmostEqual(result['gain_vs_model20'],0)
    def test_missing_or_invalid_scale(self):
        for metric in (None,SimpleNamespace(_attrs={'multiplier':0}),SimpleNamespace(_attrs={'multiplier':None})):
            self.assertNotIn('unscaled_value',score_record(SimpleNamespace(_attrs={'value':62},metric=metric)))
