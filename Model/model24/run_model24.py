"""Fixed training-size diversity study, without refitting."""
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
MODEL22 = ROOT/'models/model22/runs/f821e920af7f4f72'
MODEL22_SHA = '6b49f6a67ae9e37214b4a8142b2e15b01b152c413a96469c928dfe762d945e94'
ARMS = ('model20', 'model23', 'mix_25', 'mix_50')
CONFIG = dict(version=1, arms=list(ARMS), weights=[.25,.5], bootstrap_replicates=1000,
    bootstrap_seed=240911, confidence=.95, minimum_gain=.0003, minimum_fold_wins=4,
    interpretation='Exploratory reused-development follow-up after Model22 standalone results. '
    'Two fixed training-size probability mixtures; four simultaneous contrasts against Model20 and Model23. '
    'Conditional bootstrap excludes upstream fitting and research-selection uncertainty. No outer holdout or test labels.')

def mix_probability(base, other, weight):
    if weight not in (.25, .5):
        raise ValueError('Unspecified weight')
    prior.mix_probability(base, other)  # validate ranges, shape, finiteness
    return np.float32(1-weight)*np.asarray(base,dtype=np.float32)+np.float32(weight)*np.asarray(other,dtype=np.float32)

def initialize():
    _, upstream, _ = prior.frozen_study()
    if sha256(MODEL22/'protocol.json') != MODEL22_SHA:
        raise ValueError('Changed Model22 protocol')
    hashes = dict(upstream['contract']['hashes'])
    files = [*HERE.glob('*.py'), HERE/'readme.txt', ROOT/'tests/test_model24.py',
        MODEL22/'protocol.json', MODEL22/'summary.json']
    prediction_hashes = dict(upstream['contract']['prediction_hashes'])
    for fold in range(5):
        for suffix in ('.npz','.json','.ubj','.fit.json'):
            files.append(MODEL22/f'all_real_plus_synthetic_fold{fold}{suffix}')
        prediction_hashes[f'model22_{fold}'] = sha256(MODEL22/f'all_real_plus_synthetic_fold{fold}.npz')
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
    if arm in ARMS[:2]:
        return prior.prediction_checkpoint(directory,'model20' if arm=='model20' else 'mix_60',fold,fingerprint,held)
    base=prior.prediction_checkpoint(directory,'model20',fold,fingerprint,held)
    other=read_checkpoint(MODEL22/f'all_real_plus_synthetic_fold{fold}.npz',
        dict(protocol_sha256=MODEL22_SHA,fold=fold,arm='all_real_plus_synthetic'))
    if other is None: raise ValueError('Missing Model22 checkpoint')
    validate_predictions(other,held)
    return mix_probability(base,other['prediction'],int(arm.split('_')[1])/100)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('run','report'))
    args=parser.parse_args()
    (HERE/'runs').mkdir(parents=True,exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='run': initialize()
        from models.model24.report import report
        report()

if __name__=='__main__': main()
