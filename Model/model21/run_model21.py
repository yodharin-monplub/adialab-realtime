"""Isolated synthetic-weight study; reuse Model20 data without modifying it."""
from pathlib import Path
import argparse
import fcntl
import gc
import importlib.metadata
import json
import sys

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model20.features import block_digest, load_real_pool
from models.model20.run_model20 import synthetic_tasks, train_one
from models.model20.report import ORIGINAL_REPORT_SHA256

HERE = Path(__file__).resolve().parent
PARENT = ROOT/'models/model20/runs/82abe027caea808d'
PARENT_SHA = '4c7407df60a7be23d946b81a4029ba6ae94fc5a2fdc68648c17ffdfa99cad1ff'
ARMS = ('aligned_25', 'aligned_40', 'aligned_60')
CONFIG = dict(version=1, arms=list(ARMS), parent=str(PARENT.relative_to(ROOT)),
    parent_protocol_sha256=PARENT_SHA, fit_rows=400000, synthetic_rows=400000,
    threads=4, folds=5, baseline_parity_fold=0, bootstrap_replicates=1000,
    bootstrap_seed=210910, confidence=.95, minimum_gain=.0003, minimum_fold_wins=4,
    cloud_incumbent_tsauc=.6234481054281362, cloud_incumbent_run_id=114669,
    interpretation='Exploratory reused-development weight study, not independent confirmation. '
        'Same donors, rows, features, seeds and ensemble weights as Model20. '
        'Conditional bootstrap excludes prior selection/refitting uncertainty. '
        'Reserved holdout remains closed; no competition test labels.')


def read_cache(directory, name, expected):
    """Verify complete contiguous block coverage, then return a read-only mmap."""
    manifest = json.loads((directory/f'{name}.cache.json').read_text())
    if manifest['contract'] != expected:
        raise ValueError('Parent cache contract mismatch')
    if (manifest['completed_tasks'] != expected['tasks']
            or manifest['completed_rows'] != expected['rows']):
        raise ValueError('Parent cache incomplete')
    matrix = np.load(directory/f'{name}.npy', mmap_mode='r', allow_pickle=False)
    if matrix.shape != (expected['rows'], len(expected['names'])) or matrix.dtype != np.float32:
        raise ValueError('Parent cache shape/dtype mismatch')
    end = 0
    for block in manifest['blocks']:
        if block['start'] != end or not end <= block['end'] <= expected['rows']:
            raise ValueError('Parent cache block coverage gap or overlap')
        end = block['end']
        if block_digest(matrix[block['start']:end]) != block['sha256']:
            raise ValueError('Parent cache block checksum mismatch')
    if end != expected['rows']:
        raise ValueError('Parent cache missing blocks')
    return matrix


def initialize():
    if sha256(PARENT/'protocol.json') != PARENT_SHA:
        raise ValueError('Unexpected Model20 parent protocol')
    parent = json.loads((PARENT/'protocol.json').read_text())
    if digest(parent['plans']) != parent['plans_sha256']:
        raise ValueError('Changed parent donor plans')
    dependencies = {p: importlib.metadata.version(p) for p in
                    ('numpy', 'scipy', 'pandas', 'pyarrow', 'scikit-learn')}
    dependencies.update(xgboost=xgb.__version__, xgboost_distribution=importlib.metadata.version('xgboost-cpu'))
    if dependencies != parent['contract']['dependencies'] or sys.version != parent['contract']['python']:
        raise ValueError('Use the exact Model20 training environment')
    hashes = {}
    for relative, expected in parent['contract']['hashes'].items():
        current = sha256(ROOT/relative)
        if current != expected and not (relative == 'models/model20/report.py' and expected == ORIGINAL_REPORT_SHA256):
            raise ValueError(f'Changed Model20 source/input: {relative}')
        hashes[relative] = current
    extra = [*HERE.glob('*.py'), ROOT/'models/model20/report.py', PARENT/'protocol.json', PARENT/'summary.json',
             PARENT/'real_features.cache.json', ROOT/'models/model18/report.py']
    for fold in range(5):
        extra.append(PARENT/f'aligned_25_fold{fold}_features.cache.json')
        extra += [PARENT/f'aligned_25_fold{fold}{suffix}' for suffix in ('.json', '.npz', '.ubj', '.fit.json')]
    for path in extra:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    contract = dict(config=CONFIG, hashes=hashes, dependencies=dependencies, python=sys.version,
        recipes=parent['contract']['recipes'], names=parent['contract']['names'],
        union_names=parent['contract']['union_names'])
    directory = HERE/'runs'/digest(contract)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    protocol = dict(contract=contract, plans=parent['plans'], plans_sha256=parent['plans_sha256'])
    path = directory/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Changed Model21 frozen protocol')
    else:
        write_json(path, protocol)
    protocol_sha = sha256(path)
    write_json(HERE/'runs/latest.json', dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=protocol_sha))
    return directory, protocol, protocol_sha


def frozen_study():
    latest = json.loads((HERE/'runs/latest.json').read_text())
    directory = ROOT/latest['directory']
    protocol_sha = sha256(directory/'protocol.json')
    if protocol_sha != latest['protocol_sha256']:
        raise ValueError('Model21 protocol checksum mismatch')
    protocol = json.loads((directory/'protocol.json').read_text())
    if protocol['contract']['config'] != CONFIG or digest(protocol['plans']) != protocol['plans_sha256']:
        raise ValueError('Changed Model21 configuration or donors')
    for relative, expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Changed frozen source/input: {relative}')
    return directory, protocol, protocol_sha


def prediction_checkpoint(directory, arm, fold, protocol_sha, held):
    source, fingerprint = (PARENT, PARENT_SHA) if arm == ARMS[0] else (directory, protocol_sha)
    saved = read_checkpoint(source/f'{arm}_fold{fold}.npz',
        dict(protocol_sha256=fingerprint, fold=fold, arm=arm))
    if saved is None:
        raise RuntimeError(f'Missing {arm} fold{fold}')
    validate_predictions(saved, held)
    return saved['prediction']


def run():
    print('Freezing Model21: aligned40/aligned60 vs deployed aligned25', flush=True)
    directory, protocol, protocol_sha = initialize()
    print(f'Run directory: {directory}', flush=True)
    layout = load_development_layout(ROOT)
    contract = protocol['contract']
    real = read_cache(PARENT, 'real_features', dict(protocol_sha256=PARENT_SHA,
        rows=layout.row_count, names=contract['union_names'], tasks=int(np.sum(layout.inner_series_fold >= 0))))
    pool = load_real_pool(layout)
    for plan in protocol['plans']:
        fold = plan['fold']
        held = np.flatnonzero(layout.row_fold == fold)
        baseline = prediction_checkpoint(directory, ARMS[0], fold, protocol_sha, held)
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        sx = read_cache(PARENT, f'aligned_25_fold{fold}_features', dict(protocol_sha256=PARENT_SHA,
            rows=len(sy), names=contract['names'][fold], tasks=len(tasks)))
        if fold == CONFIG['baseline_parity_fold']:
            train_one(directory, protocol, protocol_sha, layout, real, plan, ARMS[0], (sx, sy, st))
            reproduced = read_checkpoint(directory/f'{ARMS[0]}_fold{fold}.npz',
                dict(protocol_sha256=protocol_sha, fold=fold, arm=ARMS[0]))
            validate_predictions(reproduced, held)
            np.testing.assert_array_equal(reproduced['prediction'], baseline,
                err_msg='Fresh aligned25 baseline differs; stop before new arms')
            write_json(directory/'baseline_parity.json', dict(fold=fold, observations=len(held), max_error=0.,
                parent_protocol_sha256=PARENT_SHA, protocol_sha256=protocol_sha))
            print('Aligned25 baseline reproduced bit-for-bit.', flush=True)
        for arm in ARMS[1:]:
            train_one(directory, protocol, protocol_sha, layout, real, plan, arm, (sx, sy, st))
        del sx, sy, st, tasks
        gc.collect()
    from models.model21.report import report
    report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report'))
    args = parser.parse_args()
    (HERE/'runs').mkdir(parents=True, exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Model21 is already running')
        if args.command == 'run':
            run()
        else:
            from models.model21.report import report
            report()


if __name__ == '__main__':
    main()
