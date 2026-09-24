"""Fixed equal-weight mixtures of existing synthetic-weight XGBoost voters."""
from pathlib import Path
import argparse
import fcntl
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model21.run_model21 import PARENT, PARENT_SHA

HERE = Path(__file__).resolve().parent
MODEL21 = ROOT/'models/model21/runs/35f020dc85f7912c'
MODEL21_SHA = '7a31d4213cc3d18ca260c0606c5e26ad57ac85f2ef0912052aa6e9bf6578c245'
ARMS = ('model20', 'mix_40', 'mix_60')
CONFIG = dict(version=1, arms=list(ARMS), mixture_weight=.5,
    bootstrap_replicates=1000, bootstrap_seed=230911, confidence=.95,
    minimum_gain=.0003, minimum_fold_wins=4,
    interpretation='Exploratory follow-up on reused development folds, not independent confirmation. '
        'Two fixed equal-probability mixtures: Model20 aligned25 XGBoost with Model21 aligned40 or aligned60. '
        'No weight search, new fitting or change to other ensemble members. Conditional bootstrap excludes '
        'upstream fitting and selection uncertainty. Reserved holdout stays closed; no competition test labels. '
        'Model22 continues independently and is not used to select these mixtures.')


def mix_probability(base, other):
    base, other = np.asarray(base, dtype=np.float32), np.asarray(other, dtype=np.float32)
    if base.shape != other.shape or base.ndim != 1:
        raise ValueError('Prediction vectors must have matching one-dimensional shapes')
    if not np.isfinite(base).all() or not np.isfinite(other).all():
        raise ValueError('Nonfinite mixture input')
    if np.any((base < 0) | (base > 1) | (other < 0) | (other > 1)):
        raise ValueError('Mixture inputs must be probabilities')
    return (np.float32(.5)*base+np.float32(.5)*other).astype(np.float32)


def initialize():
    if sha256(MODEL21/'protocol.json') != MODEL21_SHA or sha256(PARENT/'protocol.json') != PARENT_SHA:
        raise ValueError('Unexpected upstream studies')
    upstream = json.loads((MODEL21/'protocol.json').read_text())
    hashes = dict(upstream['contract']['hashes'])
    for relative, expected in hashes.items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Changed upstream source/input: {relative}')
    files = [*HERE.glob('*.py'), HERE/'readme.txt', ROOT/'tests/test_model23.py',
        MODEL21/'protocol.json', MODEL21/'summary.json', ROOT/'models/model21/run_model21.py']
    prediction_hashes = {}
    for arm, source in [('aligned_25', PARENT), ('aligned_40', MODEL21), ('aligned_60', MODEL21)]:
        for fold in range(5):
            for suffix in ('.npz', '.json', '.ubj', '.fit.json'):
                files.append(source/f'{arm}_fold{fold}{suffix}')
            prediction_hashes[f'{arm}_{fold}'] = sha256(source/f'{arm}_fold{fold}.npz')
    for path in files:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    protocol = dict(contract=dict(config=CONFIG, hashes=hashes, prediction_hashes=prediction_hashes),
        upstream_model21_sha256=MODEL21_SHA, upstream_model20_sha256=PARENT_SHA)
    directory = HERE/'runs'/digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Changed Model23 protocol')
    else:
        write_json(path, protocol)
    write_json(HERE/'runs/latest.json', dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=sha256(path)))
    print(f'Model23 fixed-mixture study: {directory}', flush=True)


def frozen_study():
    latest = json.loads((HERE/'runs/latest.json').read_text())
    directory = ROOT/latest['directory']
    fingerprint = sha256(directory/'protocol.json')
    protocol = json.loads((directory/'protocol.json').read_text())
    if fingerprint != latest['protocol_sha256'] or protocol['contract']['config'] != CONFIG:
        raise ValueError('Model23 protocol changed')
    for relative, expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Changed frozen source/input: {relative}')
    return directory, protocol, fingerprint


def prediction_checkpoint(directory, arm, fold, fingerprint, held):
    if arm not in ARMS:
        raise ValueError('Unspecified mixture')
    base = read_checkpoint(PARENT/f'aligned_25_fold{fold}.npz',
        dict(protocol_sha256=PARENT_SHA, fold=fold, arm='aligned_25'))
    if base is None:
        raise ValueError('Missing Model20 checkpoint')
    validate_predictions(base, held)
    if arm == ARMS[0]:
        return base['prediction']
    variant = 'aligned_'+arm.split('_')[1]
    other = read_checkpoint(MODEL21/f'{variant}_fold{fold}.npz',
        dict(protocol_sha256=MODEL21_SHA, fold=fold, arm=variant))
    if other is None:
        raise ValueError('Missing Model21 checkpoint')
    validate_predictions(other, held)
    return mix_probability(base['prediction'], other['prediction'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report'))
    args = parser.parse_args()
    (HERE/'runs').mkdir(parents=True, exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Model23 is already running')
        if args.command == 'run':
            initialize()
        from models.model23.report import report
        report()


if __name__ == '__main__':
    main()
