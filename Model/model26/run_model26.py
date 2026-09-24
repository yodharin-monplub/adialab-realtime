"""Depth diversity study using frozen real and synthetic Model20 caches."""
from pathlib import Path
import argparse
import copy
import fcntl
import gc
import importlib.metadata
import json
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model20.run_model20 import train_one, synthetic_tasks
from models.model20.features import load_real_pool
from models.model21.run_model21 import read_cache, PARENT, PARENT_SHA
from models.model23 import run_model23 as baseline
from models.model10.sources import load_development_layout

HERE = Path(__file__).resolve().parent
ARMS = ('model23', 'depth_mix')
CONFIG = dict(version=1, arms=list(ARMS), depth=4, fit_arms=['aligned_25','aligned_60'],
    bootstrap_replicates=1000, bootstrap_seed=260911, confidence=.95,
    minimum_gain=.0003, minimum_fold_wins=4,
    interpretation='Exploratory reused-development depth-diversity study. One fixed candidate; '
        'conditional bootstrap excludes upstream training and selection uncertainty. '
        'Outer holdout stays closed; no test labels. Model23 remains unchanged.')

def candidate_probability(base, p25, p60):
    return baseline.mix_probability(base, baseline.mix_probability(p25, p60))

def initialize():
    if (HERE/'runs/latest.json').exists():
        return frozen_study()
    upstream_dir, upstream, upstream_sha = baseline.frozen_study()
    parent = json.loads((PARENT/'protocol.json').read_text())
    dependencies = {p: importlib.metadata.version(p) for p in
        ('numpy','scipy','pandas','pyarrow','scikit-learn')}
    import xgboost as xgb
    dependencies.update(xgboost=xgb.__version__,
        xgboost_distribution=importlib.metadata.version('xgboost-cpu'))
    if dependencies != parent['contract']['dependencies'] or sys.version.split()[0] != parent['contract']['python'].split()[0]:
        raise ValueError('Use the exact Model20 training environment')
    hashes = dict(upstream['contract']['hashes'])
    files = [*HERE.glob('*.py'), HERE/'readme.txt', ROOT/'tests/test_model26.py',
        upstream_dir/'protocol.json', ROOT/'models/model23/run_model23.py',
        ROOT/'models/model23/report.py', ROOT/'models/model21/run_model21.py']
    for path in files:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    recipes = copy.deepcopy(parent['contract']['recipes'])
    for recipe in recipes:
        recipe['hyperparameters']['max_depth'] = CONFIG['depth']
    contract = dict(config=CONFIG, hashes=hashes, recipes=recipes,
        names=parent['contract']['names'], union_names=parent['contract']['union_names'],
        prediction_hashes=upstream['contract']['prediction_hashes'],
        baseline_protocol_sha256=upstream_sha, dependencies=dependencies, python=sys.version,
        reference_python=parent['contract']['python'],
        python_build_policy='Same patch version and dependencies; different build date requires exact baseline parity.')
    protocol = dict(contract=contract, plans=parent['plans'])
    directory = HERE/'runs'/digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol mismatch')
    if not path.exists():
        write_json(path, protocol)
    write_json(HERE/'runs/latest.json', dict(directory=str(directory.relative_to(ROOT)),
        protocol_sha256=sha256(path)))
    return directory, protocol, sha256(path)

def frozen_study():
    latest = json.loads((HERE/'runs/latest.json').read_text())
    directory = ROOT/latest['directory']
    fingerprint = sha256(directory/'protocol.json')
    protocol = json.loads((directory/'protocol.json').read_text())
    if fingerprint != latest['protocol_sha256'] or protocol['contract']['config'] != CONFIG:
        raise ValueError('Changed Model26 protocol')
    for relative, expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Changed frozen input: {relative}')
    return directory, protocol, fingerprint

def prediction_checkpoint(directory, arm, fold, fingerprint, held):
    base = baseline.prediction_checkpoint(None, 'mix_60', fold, None, held)
    if arm == 'model23':
        return base
    if arm != 'depth_mix':
        raise ValueError('Unspecified arm')
    local = []
    for fit_arm in CONFIG['fit_arms']:
        saved = read_checkpoint(directory/f'{fit_arm}_fold{fold}.npz',
            dict(protocol_sha256=fingerprint, fold=fold, arm=fit_arm))
        if saved is None:
            raise ValueError('Incomplete Model26 fits')
        validate_predictions(saved, held)
        local.append(saved['prediction'])
    return candidate_probability(base, *local)

def run():
    directory, protocol, fingerprint = initialize()
    print(f'Model26 run: {directory}', flush=True)
    layout = load_development_layout(ROOT)
    contract = protocol['contract']
    real = read_cache(PARENT, 'real_features', dict(protocol_sha256=PARENT_SHA,
        rows=layout.row_count, names=contract['union_names'],
        tasks=int(np.sum(layout.inner_series_fold >= 0))))
    pool = load_real_pool(layout)
    for plan in protocol['plans']:
        fold = plan['fold']
        held = np.flatnonzero(layout.row_fold == fold)
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        sx = read_cache(PARENT, f'aligned_25_fold{fold}_features', dict(protocol_sha256=PARENT_SHA,
            rows=len(sy), names=contract['names'][fold], tasks=len(tasks)))
        if fold == 0:
            parity = directory/'parity'
            parity.mkdir(exist_ok=True)
            original = copy.deepcopy(protocol)
            original['contract']['recipes'] = json.loads((PARENT/'protocol.json').read_text())['contract']['recipes']
            train_one(parity, original, fingerprint, layout, real, plan, 'aligned_25', (sx,sy,st))
            saved = read_checkpoint(parity/'aligned_25_fold0.npz',
                dict(protocol_sha256=fingerprint, fold=0, arm='aligned_25'))
            validate_predictions(saved, held)
            expected = baseline.prediction_checkpoint(None, 'model20', 0, None, held)
            np.testing.assert_array_equal(saved['prediction'], expected)
            write_json(directory/'baseline_parity.json', dict(max_error=0., observations=len(held)))
        for arm in CONFIG['fit_arms']:
            started = time.monotonic()
            train_one(directory, protocol, fingerprint, layout, real, plan, arm, (sx,sy,st))
            print(json.dumps(dict(fold=fold, arm=arm, elapsed_seconds=time.monotonic()-started,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
        del sx,sy,st,tasks
        gc.collect()
    from models.model26.report import report
    report()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=['run','report'])
    args=parser.parse_args()
    (HERE/'runs').mkdir(parents=True, exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command == 'run':
            run()
        else:
            from models.model26.report import report
            report()

if __name__ == '__main__':
    main()
