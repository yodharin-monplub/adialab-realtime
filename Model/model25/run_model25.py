"""Fixed rolling-peak correction study, without refitting."""
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
from models.model23 import run_model23 as prior
HERE = Path(__file__).resolve().parent
PARENT = prior.PARENT
ARMS = ('model20', 'model23', 'premium_05', 'premium_20')
CONFIG = dict(version=1, arms=list(ARMS), correction_weights=[.05,.2], bootstrap_replicates=1000,
    bootstrap_seed=250911, confidence=.95, minimum_gain=.0003, minimum_fold_wins=4,
    interpretation='Exploratory reused-development rolling-peak sensitivity study on Model23. '
    'Halve or double the existing correction; four simultaneous contrasts against Model20 and Model23. '
    'Conditional bootstrap excludes upstream fitting and research-selection uncertainty. No outer holdout or test labels.')

def initialize():
    _, upstream, _ = prior.frozen_study()
    hashes = dict(upstream['contract']['hashes'])
    files = [*HERE.glob('*.py'), HERE/'readme.txt', ROOT/'tests/test_model25.py', ROOT/'models/model20/report.py',
        ROOT/'models/model10/causal_transforms.py']
    prediction_hashes = dict(upstream['contract']['prediction_hashes'])
    for path in files:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    protocol = dict(contract=dict(config=CONFIG,hashes=hashes,prediction_hashes=prediction_hashes))
    directory = HERE/'runs'/digest(protocol)[:16]
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol mismatch')
    if not path.exists(): write_json(path,protocol)
    write_json(HERE/'runs/latest.json',dict(directory=str(directory.relative_to(ROOT)),protocol_sha256=sha256(path)))

def frozen_study():
    latest=json.loads((HERE/'runs/latest.json').read_text())
    directory=ROOT/latest['directory']
    fingerprint=sha256(directory/'protocol.json')
    protocol=json.loads((directory/'protocol.json').read_text())
    if fingerprint!=latest['protocol_sha256'] or protocol['contract']['config']!=CONFIG:
        raise ValueError('Changed protocol')
    for relative,expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/relative)!=expected: raise ValueError(f'Changed source: {relative}')
    return directory,protocol,fingerprint

def prediction_checkpoint(directory,arm,fold,fingerprint,held):
    if arm not in ARMS: raise ValueError('Unknown arm')
    return prior.prediction_checkpoint(directory,'model20' if arm=='model20' else 'mix_60',fold,fingerprint,held)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('run','report'))
    args=parser.parse_args()
    (HERE/'runs').mkdir(parents=True,exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='run': initialize()
        from models.model25.report import report
        report()

if __name__=='__main__': main()
