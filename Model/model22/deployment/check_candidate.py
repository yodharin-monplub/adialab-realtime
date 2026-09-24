"""Read-only eligibility check for LOCAL packaging, not permission to upload."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256
from models.model18.run_model18 import read_checkpoint
from models.model18.report import simultaneous_bounds
from models.model22.run_model22 import (ARMS, CONFIG, PARENT, frozen_study,
    prediction_checkpoint)
from models.model22.report import contrasts


def select_candidate(scores, lower):
    """Recompute all gates instead of trusting a saved 'passes' flag."""
    scores, lower = np.asarray(scores), np.asarray(lower)
    if scores.shape != (3, 5) or lower.shape != (3,) or not np.isfinite(scores).all() or not np.isfinite(lower).all():
        raise ValueError('Invalid final scores or confidence bounds')
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError('Scores must be normalized TS-AUC')

    def gate(a, b, bound):
        gains = scores[a]-scores[b]
        return bool(gains.mean() >= CONFIG['minimum_gain']
            and np.sum(gains > 0) >= CONFIG['minimum_fold_wins'] and bound > 0)

    passing = []
    if gate(1, 0, lower[0]):
        passing.append(1)
    if gate(2, 0, lower[1]) and gate(2, 1, lower[2]):
        passing.append(2)
    passing.sort(key=lambda i: (-scores[i].mean(), i))
    return ARMS[passing[0]] if passing else None


def verify_real_identity(control, augmented, expected_rows):
    for key in ('real_rows', 'real_rows_sha256', 'real_weights_sha256'):
        if control[key] != augmented[key]:
            raise ValueError(f'Augmentation changed {key}')
    if control['real_rows'] != expected_rows or control['synthetic_rows'] != 0:
        raise ValueError('Control did not use the frozen all-real training set')
    if augmented['synthetic_rows'] != CONFIG['synthetic_rows']:
        raise ValueError('Incorrect appended synthetic row count')
    cw, aw = control['weights'], augmented['weights']
    if aw['real_weights_unchanged'] is not True or cw['real_weights_unchanged'] is not True:
        raise ValueError('Real weights not preserved')
    if cw['real_mass'] != aw['real_mass'] or not aw['combined_mass'] > aw['real_mass']:
        raise ValueError('Synthetic weight was not added to unchanged real weight')


def check():
    directory, protocol, fingerprint = frozen_study()
    missing = [f'{arm}_fold{fold}' for arm in ARMS[1:] for fold in range(5)
        if not (directory/f'{arm}_fold{fold}.json').is_file()]
    if missing or not (directory/'summary.json').is_file():
        return dict(ready_for_local_packaging=False, reason='Research is incomplete',
            completed_fits=10-len(missing), missing_fits=missing, cloud_run_started=False)
    layout = load_development_layout(ROOT)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        for arm in ARMS:
            prediction_checkpoint(directory, arm, fold, fingerprint, held)
        records = [json.loads((directory/f'{arm}_fold{fold}.fit.json').read_text()) for arm in ARMS[1:]]
        verify_real_identity(*records, protocol['plans'][fold]['real_rows'])
    summary = json.loads((directory/'summary.json').read_text())
    if summary['model'] != 22 or summary['bootstrap_replicates'] != CONFIG['bootstrap_replicates']:
        raise ValueError('Wrong model or incomplete final bootstrap')
    metadata = dict(protocol_sha256=fingerprint, predictions={f'{arm}_{fold}':sha256(
        PARENT/f'aligned_25_fold{fold}.npz' if arm == ARMS[0] else directory/f'{arm}_fold{fold}.npz')
        for arm in ARMS for fold in range(5)})
    saved = read_checkpoint(directory/'bootstrap.npz', metadata)
    if saved is None or saved['means'].shape != (CONFIG['bootstrap_replicates'], 3):
        raise ValueError('Missing complete, verified bootstrap checkpoint')
    scores = np.asarray([summary['results'][arm]['fold_scores'] for arm in ARMS])
    lower, critical = simultaneous_bounds(contrasts(scores.mean(axis=1)), contrasts(saved['means']), CONFIG['confidence'])
    np.testing.assert_allclose(critical, summary['simultaneous_critical_value'], rtol=0, atol=1e-12)
    reported = [summary['results'][ARMS[1]]['simultaneous_lower_95'],
        summary['results'][ARMS[2]]['simultaneous_lower_95'],
        summary['results'][ARMS[2]]['vs_all_real']['simultaneous_lower_95']]
    np.testing.assert_allclose(lower, reported, rtol=0, atol=1e-12)
    candidate = select_candidate(scores, lower)
    if candidate != summary['selected_candidate']:
        raise ValueError('Saved selection disagrees with recomputed gates')
    return dict(ready_for_local_packaging=candidate is not None, selected_candidate=candidate,
        run=str(directory.relative_to(ROOT)), protocol_sha256=fingerprint,
        summary_sha256=sha256(directory/'summary.json'), completed_fits=10,
        real_data_and_weight_identity_verified=True, bootstrap_replicates=len(saved['means']),
        next_step='Packaging, streaming parity, Python3.12 smoke test, matched inference benchmark; no automatic upload',
        cloud_run_started=False)


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
