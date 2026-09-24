"""Read-only independent promotion check; never submit or modify a study."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.model24.run_model24 import frozen_study, ARMS, CONFIG
from models.model18.run_model18 import read_checkpoint

def check():
    directory, protocol, fingerprint = frozen_study()
    saved = read_checkpoint(directory/'bootstrap.npz', dict(protocol_sha256=fingerprint,
        predictions=protocol['contract']['prediction_hashes']))
    count = 0 if saved is None else len(saved['means'])
    if count != CONFIG['bootstrap_replicates'] or not (directory/'summary.json').exists():
        return dict(ready=False, reason='Evaluation incomplete', completed_bootstrap=count)
    means = np.asarray(saved['means'])
    assert means.shape == (1000,4) and np.isfinite(means).all()
    summary = json.loads((directory/'summary.json').read_text())
    preliminary = json.loads((directory/'preliminary_scores.json').read_text())
    scores = np.asarray([preliminary['ensemble_scores'][arm] for arm in ARMS])
    # Recompute contrasts explicitly, independently of report.contrasts/passes.
    observed = np.array([scores[i].mean()-scores[j].mean() for i in (2,3) for j in (0,1)])
    bootstrap = np.column_stack([means[:,i]-means[:,j] for i in (2,3) for j in (0,1)])
    from models.model18.report import simultaneous_bounds
    lower, critical = simultaneous_bounds(observed, bootstrap, .95)
    passing=[]
    for i in (2,3):
        passed=True
        for j in (0,1):
            k=(i-2)*2+j
            wins=int(np.sum(scores[i]>scores[j]))
            expected=bool(observed[k]>=.0003 and wins>=4 and lower[k]>0)
            result=summary['results'][ARMS[i]]['comparisons'][ARMS[j]]
            assert result['passes']==expected and result['fold_wins']==wins
            np.testing.assert_allclose(result['simultaneous_lower_95'],lower[k],rtol=0,atol=1e-12)
            passed &= expected
        assert summary['results'][ARMS[i]]['passes']==passed
        if passed: passing.append(ARMS[i])
    passing.sort(key=lambda arm: (-scores[ARMS.index(arm)].mean(),int(arm.split('_')[1])))
    selected=passing[0] if passing else None
    assert summary['selected_candidate']==selected
    return dict(ready=selected is not None, selected_candidate=selected,
        completed_bootstrap=count, hashes_verified=True, gates_independently_verified=True,
        lower_bounds=lower.tolist(), reason='Local gate only; not upload authorization')

if __name__=='__main__':
    print(json.dumps(check(),indent=2))
