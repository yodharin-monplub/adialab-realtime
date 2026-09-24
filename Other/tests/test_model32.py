import json
import unittest
import numpy as np
from models.model32.run_model32 import CONFIG,meta_features

class Model32Tests(unittest.TestCase):
    def test_meta_features_exact_width(self):
        x=meta_features(np.asarray([-.2,.4]),np.asarray([.2,.8]),np.asarray([.3,.7]),np.asarray([0,9]))
        self.assertEqual(x.shape,(2,4));self.assertTrue(np.isfinite(x).all())
    def test_crossfit_contract(self):
        self.assertEqual(CONFIG["folds"],[0,1,2,3,4]);self.assertIn("other four",CONFIG["meta_training"])
    def test_upstream_hashes_are_explicit(self):
        self.assertEqual(len(CONFIG["model27_protocol_sha256"]),64);self.assertEqual(len(CONFIG["model30_protocol_sha256"]),64)

if __name__=="__main__":unittest.main()
