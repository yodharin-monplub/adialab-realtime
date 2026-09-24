"""End-to-end reporting smoke test on tiny, entirely artificial fixtures."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from models.model22 import report as reporting
from tsauc import tsauc


class Model22ReportTest(unittest.TestCase):
    def test_report_bootstrap_resume_and_joint_promotion(self):
        # Twenty whole series per fold, with both classes at each online step.
        series = np.repeat(np.arange(100), 4)
        folds = series//20
        target = (series % 2).astype(np.uint8)
        steps = np.tile(np.arange(4), 100)
        layout = SimpleNamespace(row_count=len(series), target=target, time_online=steps,
            series_position=series, row_fold=folds, inner_series_fold=np.arange(100)//20,
            tau_index=np.where(np.arange(100) % 2 == 0, -1, 0))
        # Baseline is tied. Full-real gives partial separation. Augmentation
        # perfectly separates this artificial fixture; not a research result.
        baseline = np.full(len(series), .5, dtype=np.float32)
        real = np.where(series % 20 < 16, .5, np.where(target, .6, .4)).astype(np.float32)
        augmented = np.where(target, .8, .2).astype(np.float32)
        vectors = dict(zip(reporting.ARMS, (baseline, real, augmented)))
        expected = [tsauc(target[folds == f], baseline[folds == f], steps[folds == f]) for f in range(5)]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, here, directory = root/'parent', root/'model22', root/'run'
            for path in (parent, here, directory):
                path.mkdir()
            (parent/'summary.json').write_text(json.dumps(dict(results=dict(aligned_25=dict(
                ensemble_vs_incumbent=dict(fold_scores=expected))))))

            def prediction(directory, arm, fold, fingerprint, held):
                return vectors[arm][held]

            with (patch.object(reporting, 'PARENT', parent), patch.object(reporting, 'HERE', here),
                  patch.object(reporting, 'frozen_study', return_value=(directory, {}, 'fixture')),
                  patch.object(reporting, 'load_development_layout', return_value=layout),
                  patch.object(reporting, 'prediction_checkpoint', side_effect=prediction),
                  patch.object(reporting, 'ensemble_margin', side_effect=lambda probability, layout: probability),
                  patch.object(reporting, 'sha256', side_effect=lambda path: str(path)),
                  patch.dict(reporting.CONFIG, bootstrap_replicates=50)):
                summary = reporting.report()
                self.assertEqual(summary['model'], 22)
                self.assertEqual(summary['bootstrap_replicates'], 50)
                self.assertEqual(summary['selected_candidate'], 'all_real_plus_synthetic')
                result = summary['results']['all_real_plus_synthetic']
                self.assertTrue(result['passes'])
                self.assertGreater(result['vs_all_real']['mean_gain'], 0)
                self.assertGreater(result['vs_all_real']['simultaneous_lower_95'], 0)
                self.assertEqual(json.loads((directory/'summary.json').read_text()), summary)
                self.assertTrue((here/'RESULTS.md').is_file())
                cached = (directory/'bootstrap.npz').read_bytes()
                # Completed bootstrap must load without constructing scorers.
                with patch.object(reporting, 'WeightedTSAUC', side_effect=AssertionError('Should reuse checkpoint')):
                    resumed = reporting.report()
                self.assertEqual(resumed, summary)
                self.assertEqual(cached, (directory/'bootstrap.npz').read_bytes())
                # Beating Model20 alone is insufficient to claim augmentation
                # helped: the all-real comparison must independently pass.
                with patch.object(reporting, 'simultaneous_bounds',
                        return_value=(np.array([.1, .1, -.001]), .2)):
                    rejected = reporting.report()
                self.assertFalse(rejected['results']['all_real_plus_synthetic']['passes'])
                self.assertEqual(rejected['selected_candidate'], 'all_real')


if __name__ == '__main__':
    unittest.main()
