import unittest
import numpy as np
from models.model30.run_model30 import CONFIG,candidate_probability

class Model30Tests(unittest.TestCase):
    def test_exact_float32_blend(self):
        a=np.asarray([.1,.9],dtype=np.float32); b=np.asarray([.7,.3],dtype=np.float32)
        expected=np.float32(.5)*a+np.float32(.5)*b
        actual=candidate_probability(a,b)
        self.assertEqual(actual.dtype,np.float32); np.testing.assert_array_equal(actual,expected)
    def test_lower_shift_contract(self):
        self.assertEqual(CONFIG["synthetic_share"],.15); self.assertEqual(CONFIG["depth"],3)

if __name__=="__main__": unittest.main()
