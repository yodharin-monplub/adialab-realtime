import unittest
import numpy as np
from models.model24.run_model24 import mix_probability, ARMS
from models.model24.report import contrasts, passes

class Model24Test(unittest.TestCase):
    def test_mixture(self):
        a=np.array([0,.4,1],dtype=np.float32)
        b=np.array([1,.8,0],dtype=np.float32)
        np.testing.assert_allclose(mix_probability(a,b,.25),[.25,.5,.75])
        np.testing.assert_array_equal(mix_probability(a,b,.5),mix_probability(b,a,.5))
        self.assertEqual(mix_probability(a,b,.25).dtype,np.float32)
        for invalid in (0,.3,1):
            with self.assertRaises(ValueError): mix_probability(a,b,invalid)
        with self.assertRaises(ValueError): mix_probability([np.nan],[.2],.25)
    def test_contrasts_and_gate(self):
        self.assertEqual(ARMS,('model20','model23','mix_25','mix_50'))
        np.testing.assert_array_equal(contrasts(np.array([[1,2,4,8]])),[[3,2,7,6]])
        self.assertTrue(passes(dict(mean_gain=.0003,fold_wins=4,simultaneous_lower_95=.00001)))
        self.assertFalse(passes(dict(mean_gain=.0003,fold_wins=4,simultaneous_lower_95=-.00001)))

if __name__=='__main__': unittest.main()
