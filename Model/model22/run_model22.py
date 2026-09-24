"""All eligible real rows, with or without appended aligned synthetic rows."""
from pathlib import Path
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, commit_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import reference_weights, synthetic_tasks
from models.model21.run_model21 import PARENT, PARENT_SHA, read_cache
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
AUDIT = ROOT/'models/model21/runs/35f020dc85f7912c/protocol.json'
AUDIT_SHA = '7a31d4213cc3d18ca260c0606c5e26ad57ac85f2ef0912052aa6e9bf6578c245'
ARMS = ('model20', 'all_real', 'all_real_plus_synthetic')
CONFIG = dict(version=1, arms=list(ARMS), folds=5, threads=4,
    real_sampling='all eligible rows; no sampling or cap; preserve validation/stopping roles',
    synthetic_rows=400000, synthetic_weights='same real-reference (step,target) weight per row; unsupported cells zero',
    real_weights='computed on all real fitting rows once; unchanged by augmentation; no combined renormalization',
    bootstrap_replicates=1000, bootstrap_seed=220911, confidence=.95,
    minimum_gain=.0003, minimum_fold_wins=4,
    interpretation='Exploratory reused-development experiment. All real fitting rows retained. '
        'Synthetic rows appended without reducing real weights. Same donors, features, seeds and '
        'ensemble settings as Model20. Conditional bootstrap excludes prior selection/refitting '
        'uncertainty. Reserved outer holdout stays closed; no competition test labels.',
    incumbent_cloud_tsauc=.6234481054281362)


def all_fitting_rows(layout, fold):
    return np.flatnonzero((layout.row_fold >= 0) & (layout.row_fold != fold)
        & (layout.row_fold != (fold+1) % 5))


def row_digest(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def additive_weights(y, t, sy, st):
    """Real weights are immutable; synthetic rows inherit real cell weights."""
    real = reference_weights(y, t, y, t)
    rc = 2*np.asarray(t, dtype=np.int64)+y
    sc = 2*np.asarray(st, dtype=np.int64)+sy
    size = int(max(rc.max(initial=0), sc.max(initial=0)))+1
    lookup = np.zeros(size, dtype=np.float32)
    lookup[rc] = real  # Every real row in the same cell has the same weight.
    synthetic = lookup[sc]
    real_mass = float(real.sum(dtype=np.float64))
    synthetic_mass = float(synthetic.sum(dtype=np.float64))
    return real, synthetic, dict(real_mass=real_mass, synthetic_mass=synthetic_mass,
        combined_mass=real_mass+synthetic_mass,
        synthetic_weight_share=synthetic_mass/(real_mass+synthetic_mass) if real_mass+synthetic_mass else 0.,
        real_weights_unchanged=True, synthetic_zero_weight_rows=int(np.sum(synthetic == 0)))


def initialize(layout):
    if sha256(AUDIT) != AUDIT_SHA or sha256(PARENT/'protocol.json') != PARENT_SHA:
        raise ValueError('Unexpected parent provenance')
    audit = json.loads(AUDIT.read_text())
    source = audit['contract']
    dependencies = {p: importlib.metadata.version(p) for p in
        ('numpy', 'scipy', 'pandas', 'pyarrow', 'scikit-learn')}
    dependencies.update(xgboost=xgb.__version__, xgboost_distribution=importlib.metadata.version('xgboost-cpu'))
    if dependencies != source['dependencies'] or sys.version != source['python']:
        raise ValueError('Use the same training environment as Model20/21')
    hashes = dict(source['hashes'])
    for relative, expected in hashes.items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Changed parent source/input: {relative}')
    for path in [AUDIT, *HERE.glob('*.py'), HERE/'readme.txt',
                 ROOT/'models/model21/run_model21.py', ROOT/'tests/test_model22.py']:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    parent = json.loads((PARENT/'protocol.json').read_text())
    if digest(parent['plans']) != parent['plans_sha256']:
        raise ValueError('Changed donor plans')
    plans = []
    for old in parent['plans']:
        rows = all_fitting_rows(layout, old['fold'])
        plan = {k:v for k,v in old.items() if k != 'fit_rows'}
        plan.update(real_rows=len(rows), real_rows_sha256=row_digest(rows))
        plans.append(plan)
    contract = dict(config=CONFIG, hashes=hashes, dependencies=dependencies, python=sys.version,
        recipes=source['recipes'], names=source['names'], union_names=source['union_names'])
    protocol = dict(contract=contract, plans=plans, plans_sha256=digest(plans))
    directory = HERE/'runs'/digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Changed Model22 protocol')
    else:
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE/'runs/latest.json', dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=fingerprint))
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE/'runs/latest.json').read_text())
    directory = ROOT/latest['directory']
    fingerprint = sha256(directory/'protocol.json')
    protocol = json.loads((directory/'protocol.json').read_text())
    if fingerprint != latest['protocol_sha256'] or protocol['contract']['config'] != CONFIG:
        raise ValueError('Model22 frozen protocol changed')
    if digest(protocol['plans']) != protocol['plans_sha256']:
        raise ValueError('Model22 plans changed')
    for relative, expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Changed frozen source/input: {relative}')
    return directory, protocol, fingerprint


def prediction_checkpoint(directory, arm, fold, fingerprint, held):
    source, tag, name = (PARENT, PARENT_SHA, 'aligned_25') if arm == ARMS[0] else (directory, fingerprint, arm)
    saved = read_checkpoint(source/f'{name}_fold{fold}.npz', dict(protocol_sha256=tag, fold=fold, arm=name))
    if saved is None:
        raise RuntimeError(f'Missing {arm} fold{fold}')
    validate_predictions(saved, held)
    return saved['prediction']


def make_matrix(real, rows, columns, synthetic=None):
    """Avoid materializing the much wider union-feature training matrix."""
    count = len(rows)+(0 if synthetic is None else len(synthetic))
    matrix = np.empty((count, len(columns)), dtype=np.float32)
    for left in range(0, len(rows), 25000):
        selected = rows[left:left+25000]
        matrix[left:left+len(selected)] = real[np.ix_(selected, columns)]
    if synthetic is not None:
        matrix[len(rows):] = synthetic
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite training features')
    return matrix


def train_one(directory, protocol, fingerprint, layout, real, plan, arm, synthetic):
    fold = plan['fold']
    held = np.flatnonzero(layout.row_fold == fold)
    path = directory/f'{arm}_fold{fold}.npz'
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm=arm)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        return
    started = time.monotonic()
    rows = all_fitting_rows(layout, fold)
    if len(rows) != plan['real_rows'] or row_digest(rows) != plan['real_rows_sha256']:
        raise ValueError('Full real row selection differs from frozen plan')
    y, t = layout.target[rows], layout.time_online[rows]
    sx, sy, st = synthetic
    real_weights, synthetic_weights, accounting = additive_weights(y, t, sy, st)
    np.testing.assert_array_equal(real_weights, reference_weights(y, t, y, t))
    use_synthetic = arm == ARMS[2]
    if arm not in ARMS[1:]:
        raise ValueError('Unknown training arm')
    weights = np.r_[real_weights, synthetic_weights] if use_synthetic else real_weights
    np.testing.assert_array_equal(weights[:len(rows)], real_weights)
    target = np.r_[y, sy] if use_synthetic else y
    contract = protocol['contract']
    columns = np.asarray([contract['union_names'].index(name) for name in contract['names'][fold]])
    matrix = make_matrix(real, rows, columns, sx if use_synthetic else None)
    recipe = contract['recipes'][fold]
    params = {k:v for k,v in recipe['hyperparameters'].items() if k != 'name'}
    model = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', n_estimators=recipe['rounds'],
        eval_metric='logloss', random_state=recipe['seed'], n_jobs=CONFIG['threads'], device='cpu', verbosity=0, **params)
    print(f'Fitting {arm} fold{fold}: {len(rows):,} real + {len(sy) if use_synthetic else 0:,} synthetic rows; '
        f'{matrix.nbytes/1024**3:.2f} GiB feature matrix; {recipe["rounds"]} trees', flush=True)
    model.fit(matrix, target, sample_weight=weights, verbose=False)
    del matrix
    gc.collect()
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 25000):
        selected = held[left:left+25000]
        prediction[left:left+len(selected)] = model.predict_proba(real[np.ix_(selected, columns)])[:, 1]
    model_path = directory/f'{arm}_fold{fold}.ubj'
    temporary = directory/f'{arm}_fold{fold}.partial.ubj'
    model.save_model(temporary)
    temporary.replace(model_path)
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    if not use_synthetic:
        accounting.update(synthetic_mass=0., combined_mass=accounting['real_mass'], synthetic_weight_share=0.,
            synthetic_zero_weight_rows=0)
    details = directory/f'{arm}_fold{fold}.fit.json'
    write_json(details, dict(score=score, real_rows=len(rows), synthetic_rows=len(sy) if use_synthetic else 0,
        real_rows_sha256=row_digest(rows), real_weights_sha256=hashlib.sha256(real_weights.tobytes()).hexdigest(),
        weights=accounting, elapsed_seconds=time.monotonic()-started))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta, (model_path, details))
    print(f'Finished {arm} fold{fold}: component TSAUC={score:.9f}; {time.monotonic()-started:.1f}s', flush=True)
    del model
    gc.collect()


def run():
    layout = load_development_layout(ROOT)
    directory, protocol, fingerprint = initialize(layout)
    print(f'Model22 run: {directory}', flush=True)
    print('All-real counts by fold: '+str([p['real_rows'] for p in protocol['plans']]), flush=True)
    real = read_cache(PARENT, 'real_features', dict(protocol_sha256=PARENT_SHA, rows=layout.row_count,
        names=protocol['contract']['union_names'], tasks=layout.series_count))
    pool = load_real_pool(layout)
    for plan in protocol['plans']:
        fold = plan['fold']
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        if len(sy) != CONFIG['synthetic_rows']:
            raise ValueError('Unexpected synthetic row count')
        sx = read_cache(PARENT, f'aligned_25_fold{fold}_features', dict(protocol_sha256=PARENT_SHA,
            rows=len(sy), names=protocol['contract']['names'][fold], tasks=len(tasks)))
        for arm in ARMS[1:]:
            train_one(directory, protocol, fingerprint, layout, real, plan, arm, (sx, sy, st))
        before, after = [json.loads((directory/f'{arm}_fold{fold}.fit.json').read_text()) for arm in ARMS[1:]]
        for key in ('real_rows', 'real_rows_sha256', 'real_weights_sha256'):
            if before[key] != after[key]:
                raise ValueError('Augmentation changed real data or weights')
        del sx, sy, st, tasks
        gc.collect()
    from models.model22.report import report
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
            raise SystemExit('Model22 is already running')
        if args.command == 'run':
            run()
        else:
            from models.model22.report import report
            report()


if __name__ == '__main__':
    main()
