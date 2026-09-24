import unittest
import numpy as np
from types import SimpleNamespace
from models.model20.run_model20 import synthetic_tasks
from models.model26.run_model26 import candidate_probability

class Model26Test(unittest.TestCase):
    def test_donors_exclude_validation_stopping_and_holdout(self):
        for role in (-1,0,1):
            layout=SimpleNamespace(inner_series_fold=np.array([role,2]))
            plan=dict(fold=0, selected=[], pairs=[dict(a=0,b=1)])
            with self.assertRaisesRegex(ValueError,'donor leaks'):
                synthetic_tasks({},plan,layout,True)
    def test_nested_float32_mixture(self):
        a=np.array([0.,1.,.13], dtype=np.float32)
        b=np.array([1.,0.,.87], dtype=np.float32)
        c=np.array([.2,.8,.33], dtype=np.float32)
        expected=np.float32(.5)*a+np.float32(.5)*(np.float32(.5)*b+np.float32(.5)*c)
        np.testing.assert_array_equal(candidate_probability(a,b,c), expected)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            candidate_probability([.5],[float('nan')],[.5])
