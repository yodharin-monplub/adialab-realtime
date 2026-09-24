import json
import unittest
from pathlib import Path
import numpy as np
from models.model31.run_model31 import CONFIG,meta_features

class Model31Tests(unittest.TestCase):
    def test_features_follow_prespec(self):
        result=meta_features(np.asarray([-.2,.4]),np.asarray([.3,.8]),np.asarray([0,9]))
        self.assertEqual(result.shape,(2,3)); self.assertTrue(np.isfinite(result).all())
    def test_disjoint_confirmation(self):
        self.assertEqual(CONFIG["pilot_folds"],[0]); self.assertEqual(CONFIG["confirmation_folds"],[1,2,3,4])
        self.assertFalse(set(CONFIG["pilot_folds"])&set(CONFIG["confirmation_folds"]))
    def test_prespec_names_exact_inputs(self):
        self.assertEqual(CONFIG["features"],["model26_final_margin","model30_raw_logit","log1p_time"])

if __name__=="__main__": unittest.main()
