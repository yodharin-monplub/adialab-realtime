"""Read-only verification of Model26's frozen evidence and promotion gates."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.model26.run_model26 import frozen_study, ARMS
from models.model11.pipeline import sha256
from models.model18.run_model18 import read_checkpoint
from models.model18.report import simultaneous_bounds


def check():
    directory, protocol, fingerprint = frozen_study()
    predictions = sorted(directory.glob('aligned_*_fold*.npz'))
    assert len(predictions) == 10
    saved = read_checkpoint(directory / 'bootstrap.npz', dict(
        protocol_sha256=fingerprint,
        predictions={p.name: sha256(p) for p in predictions}))
    assert saved is not None
    means = np.asarray(saved['means'])
    assert means.shape == (1000, 2) and np.isfinite(means).all()
    preliminary = json.loads((directory / 'preliminary_scores.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    scores = np.asarray([preliminary['ensemble_scores'][arm] for arm in ARMS])
    assert scores.shape == (2, 5) and np.isfinite(scores).all()
    gain = float(scores[1].mean() - scores[0].mean())
    wins = int(np.sum(scores[1] > scores[0]))
    lower, critical = simultaneous_bounds(np.array([gain]),
        (means[:, 1] - means[:, 0])[:, None], .95)
    passed = bool(gain >= .0003 and wins >= 4 and lower[0] > 0)
    result = summary['results']['depth_mix']
    np.testing.assert_allclose(result['fold_scores'], scores[1], rtol=0, atol=0)
    np.testing.assert_allclose(result['mean_gain'], gain, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['simultaneous_lower_95'], lower[0], rtol=0, atol=1e-12)
    assert result['passes'] == passed and result['fold_wins'] == wins
    assert summary['selected_candidate'] == ('depth_mix' if passed else None)
    return dict(ready=passed, selected_candidate=summary['selected_candidate'],
        protocol_sha256=fingerprint, completed_bootstrap=1000,
        hashes_verified=True, gates_independently_recomputed=True,
        cv=float(scores[1].mean()), mean_gain=gain, fold_wins=wins,
        lower_95=float(lower[0]), caveat='Reused development CV; not a cloud result or upload authorization.')


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
