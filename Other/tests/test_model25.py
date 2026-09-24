import unittest
import numpy as np
from models.model25.postprocess import correction_weight
from models.model25.report import contrasts, passes
from models.model25.run_model25 import ARMS

class Model25Test(unittest.TestCase):
    def test_fixed_strengths(self):
        self.assertEqual([correction_weight(a) for a in ARMS],[.1,.1,.05,.2])
        with self.assertRaises(KeyError): correction_weight('premium_30')
    def test_joint_gate(self):
        np.testing.assert_array_equal(contrasts(np.array([1,2,4,8])),[3,2,7,6])
        self.assertFalse(passes(dict(mean_gain=.0002,fold_wins=5,simultaneous_lower_95=.0001)))

if __name__=='__main__': unittest.main()
