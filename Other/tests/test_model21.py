import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from models.model20.features import block_digest
from models.model20.run_model20 import splice_weights
from models.model21.run_model21 import ARMS, CONFIG, read_cache
from models.model21.report import contrasts, passes


class Model21Test(unittest.TestCase):
    def test_fixed_arms_and_contrasts(self):
        self.assertEqual(ARMS, ('aligned_25', 'aligned_40', 'aligned_60'))
        self.assertEqual(CONFIG['fit_rows'], 400000)
        self.assertEqual(CONFIG['synthetic_rows'], 400000)
        np.testing.assert_allclose(contrasts([.63, .64, .62]), [.01, -.01])
        np.testing.assert_array_equal(contrasts([[1, 2, 3], [4, 6, 8]]), [[1, 2], [2, 4]])

    def test_higher_weights_preserve_mass(self):
        y = np.tile([0, 1], 5).astype(np.uint8)
        t = np.repeat(np.arange(5), 2)
        weights = np.arange(1, 11, dtype=np.float32)
        for share in (.4, .6):
            rw, sw, stats = splice_weights(weights, y, t, y[:8], t[:8], share)
            after = np.bincount(2*t+y, weights=rw)+np.bincount(2*t[:8]+y[:8], weights=sw, minlength=10)
            np.testing.assert_allclose(after, weights, rtol=2e-7)
            np.testing.assert_array_equal(rw[8:], weights[8:])
            self.assertLess(stats['actual_share'], share)
            self.assertTrue(np.all(rw >= 0) and np.all(sw >= 0))

    def test_gate_requires_all_conditions_and_is_json_safe(self):
        good = dict(mean_gain=np.float64(.0004), fold_wins=4, simultaneous_lower_95=np.float64(.0001))
        self.assertIs(passes(good), True)
        json.dumps(dict(passes=passes(good)))
        for key, value in [('mean_gain', .0002), ('fold_wins', 3), ('simultaneous_lower_95', 0)]:
            self.assertIs(passes({**good, key:value}), False)

    def test_cache_readonly_contract_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = np.arange(12, dtype=np.float32).reshape(6, 2)
            np.save(directory/'x.npy', data)
            expected = dict(protocol_sha256='p', rows=6, names=['a', 'b'], tasks=2)
            manifest = dict(contract=expected, completed_rows=6, completed_tasks=2,
                blocks=[dict(start=0, end=3, sha256=block_digest(data[:3])),
                        dict(start=3, end=6, sha256=block_digest(data[3:]))])
            path = directory/'x.cache.json'
            path.write_text(json.dumps(manifest))
            before = (directory/'x.npy').read_bytes()
            matrix = read_cache(directory, 'x', expected)
            np.testing.assert_array_equal(matrix, data)
            self.assertFalse(matrix.flags.writeable)
            self.assertEqual(before, (directory/'x.npy').read_bytes())
            with self.assertRaisesRegex(ValueError, 'contract'):
                read_cache(directory, 'x', {**expected, 'protocol_sha256':'wrong'})
            manifest['blocks'].pop()
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'missing blocks'):
                read_cache(directory, 'x', expected)
            manifest['blocks'] = [dict(start=0, end=6, sha256='wrong')]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_cache(directory, 'x', expected)


if __name__ == '__main__':
    unittest.main()
