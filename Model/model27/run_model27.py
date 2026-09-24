"""Matched contrastive augmentation and history-context gating study."""
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
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, commit_checkpoint, read_checkpoint, validate_predictions
from models.model20.features import build_cache, load_real_pool
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights
from models.model21.run_model21 import read_cache, PARENT, PARENT_SHA
from models.model26 import run_model26 as baseline
from models.model27.augmentation import (CONTEXT_NAMES, contexts,
    select_matched_pairs, assemble)
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
ARMS = ('model26', 'matched_context')
CONFIG = dict(version=1, arms=list(ARMS), seed=270911, pairs=1000,
    positive_rows=200_000, negative_rows=200_000, synthetic_share=.25,
    donor_candidates=256, max_b_reuse=5, context_names=list(CONTEXT_NAMES),
    depth=4, rounds=800, incumbent_weight=.8, candidate_weight=.2,
    bootstrap_replicates=1000, bootstrap_seed=271109, confidence=.95,
    target_tsauc=.64, minimum_gain=.0007, minimum_fold_wins=4,
    interpretation='Prespecified family-matched contrastive augmentation. Every new fit excludes its '
        'held series and the adjacent stopping fold. Positive and hard-negative composites share seam '
        'construction. History context is inference-safe. Existing feature selection and ensemble weights '
        'remain development-informed; outer holdout and test labels remain closed.')


def candidate_probability(base, new):
    base = np.asarray(base, dtype=np.float32)
    new = np.asarray(new, dtype=np.float32)
    if base.shape != new.shape or not np.isfinite(base).all() or not np.isfinite(new).all():
        raise ValueError('Invalid candidate probabilities')
    return np.float32(CONFIG['incumbent_weight'])*base + np.float32(CONFIG['candidate_weight'])*new


def source_files(upstream_dir):
    files = [*HERE.glob('*.py'), HERE/'readme.txt', ROOT/'tests/test_model27.py',
        upstream_dir/'protocol.json', ROOT/'models/model26/run_model26.py',
        ROOT/'models/model26/report.py', ROOT/'models/model20/run_model20.py',
        ROOT/'models/model20/features.py', ROOT/'models/model21/run_model21.py',
        ROOT/'models/model10/sources.py', ROOT/'models/model11/pipeline.py',
        ROOT/'models/model18/run_model18.py', ROOT/'models/model18/report.py',
        ROOT/'models/model20/report.py', ROOT/'models/report_next_models.py', ROOT/'tsauc.py']
    return sorted(set(files))


def initialize(layout, pool):
    if (HERE/'runs/latest.json').exists():
        return frozen_study()
    upstream_dir, upstream, upstream_sha = baseline.frozen_study()
    parent = json.loads((PARENT/'protocol.json').read_text())
    dependencies = {p:importlib.metadata.version(p) for p in
        ('numpy','scipy','pandas','pyarrow','scikit-learn')}
    dependencies.update(xgboost=xgb.__version__,
        xgboost_distribution=importlib.metadata.version('xgboost-cpu'))
    if dependencies != parent['contract']['dependencies'] or sys.version.split()[0] != parent['contract']['python'].split()[0]:
        raise ValueError('Use the exact Model20/26 training environment')
    recipes = copy.deepcopy(parent['contract']['recipes'])
    for recipe in recipes:
        recipe['hyperparameters']['max_depth'] = CONFIG['depth']
        recipe['rounds'] = CONFIG['rounds']
    history_context = contexts(pool)
    plans = []
    for fold, recipe in enumerate(recipes):
        fit, _ = fitting_rows(layout, fold, recipe['seed'])
        pairs, stats, _ = select_matched_pairs(pool, layout.tau_index,
            layout.inner_series_fold, fold, (fold+1)%5, CONFIG['seed']+fold,
            count=CONFIG['pairs'], reuse=CONFIG['max_b_reuse'],
            candidates=CONFIG['donor_candidates'], context=history_context)
        if len(pairs) != CONFIG['pairs']:
            raise ValueError(f'Insufficient matched pairs in fold {fold}: {stats}')
        total = sum(p['length'] for p in pairs)
        selections = {}
        for branch, rows, stream in [('positive', CONFIG['positive_rows'], 1),
                                     ('negative', CONFIG['negative_rows'], 2)]:
            selections[branch] = np.sort(np.random.default_rng(
                np.random.SeedSequence([CONFIG['seed']+fold, stream])).choice(
                    total, min(total, rows), replace=False)).tolist()
        plans.append(dict(fold=fold, fit_rows=fit.tolist(), pairs=pairs,
            stats=stats, synthetic_total_per_branch=total, selections=selections))
        print(f'Frozen matched fold {fold}: {stats}; branch rows={len(selections["positive"]):,}', flush=True)
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in source_files(upstream_dir)}
    contract = dict(config=CONFIG, hashes=hashes, recipes=recipes,
        names=parent['contract']['names'], union_names=parent['contract']['union_names'],
        dependencies=dependencies, python=sys.version, baseline_protocol_sha256=upstream_sha,
        research=['Adams and MacKay 2007 BOCPD', 'Puchkin et al. 2023 contrastive online CPD'])
    protocol = dict(contract=contract, plans=plans)
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
    protocol = json.loads((directory/'protocol.json').read_text())
    fingerprint = sha256(directory/'protocol.json')
    if fingerprint != latest['protocol_sha256'] or protocol['contract']['config'] != CONFIG:
        raise ValueError('Changed Model27 protocol')
    for relative, expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Changed frozen input: {relative}')
    return directory, protocol, fingerprint


def synthetic_tasks(pool, context, plan, layout):
    tasks, targets, times, row_context = [], [], [], []
    for branch, changed in [('positive', True), ('negative', False)]:
        selected = np.asarray(plan['selections'][branch], dtype=np.int64)
        offset = 0
        for pair in plan['pairs']:
            for key in ('a','b'):
                if pair[key] not in pool:
                    raise ValueError('Unknown donor')
                if pair[key] not in context:
                    raise ValueError('Missing donor context')
                if layout.inner_series_fold[pair[key]] in (-1, plan['fold'], (plan['fold']+1)%5):
                    raise ValueError('Synthetic donor leaks held, stopping, or reserved series')
            h, o, y, t = assemble(pool, pair, changed)
            left, right = np.searchsorted(selected, [offset, offset+len(o)])
            keep = selected[left:right]-offset
            tasks.append((h, o, keep))
            targets.append(y[keep]); times.append(t[keep])
            row_context.append(np.repeat(context[pair['a']][None,:], len(keep), axis=0))
            offset += len(o)
    return tasks, np.concatenate(targets), np.concatenate(times), np.concatenate(row_context)


def fit_one(directory, protocol, fingerprint, layout, real, context_matrix, plan, synthetic):
    fold = plan['fold']; arm = 'matched_context'
    path = directory/f'{arm}_fold{fold}.npz'
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm=arm)
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        return
    recipe = protocol['contract']['recipes'][fold]
    names = protocol['contract']['names'][fold]
    union = protocol['contract']['union_names']
    columns = np.asarray([union.index(n) for n in names])
    fit, fitting = fitting_rows(layout, fold, recipe['seed'])
    np.testing.assert_array_equal(fit, plan['fit_rows'])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    sx, sy, st, scontext = synthetic
    rw, sw, accounting = splice_weights(weights, y, t, sy, st, CONFIG['synthetic_share'])
    base_real = np.asarray(real[fit][:,columns], dtype=np.float32)
    real_context = context_matrix[layout.series_position[fit]]
    x = np.concatenate((np.concatenate((base_real, real_context), axis=1),
        np.concatenate((np.asarray(sx), scontext), axis=1)), axis=0)
    y = np.r_[y,sy]; weights = np.r_[rw,sw]
    params = {k:v for k,v in recipe['hyperparameters'].items() if k != 'name'}
    model = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist',
        n_estimators=recipe['rounds'], eval_metric='logloss', random_state=recipe['seed'],
        n_jobs=4, device='cpu', verbosity=0, **params)
    started = time.monotonic()
    print(f'Fitting Model27 fold {fold}: {len(y):,} rows, {x.shape[1]} features', flush=True)
    model.fit(x,y,sample_weight=weights,verbose=False)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0,len(held),50_000):
        rows = held[left:left+50_000]
        hx = np.concatenate((np.asarray(real[rows][:,columns], dtype=np.float32),
            context_matrix[layout.series_position[rows]]), axis=1)
        prediction[left:left+len(rows)] = model.predict_proba(hx)[:,1]
    model_path=directory/f'{arm}_fold{fold}.ubj'; temporary=model_path.with_suffix('.partial.ubj')
    model.save_model(temporary); temporary.replace(model_path)
    details=directory/f'{arm}_fold{fold}.fit.json'
    write_json(details,dict(component_score=tsauc(layout.target[held],prediction,layout.time_online[held]),
        real_rows=len(fit),synthetic_rows=len(sy),synthetic_label_one_rows=int(sy.sum()),
        weights=accounting,elapsed_seconds=time.monotonic()-started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    commit_checkpoint(path,dict(prediction=prediction,held_rows=held),meta,(model_path,details))
    print(f'Finished Model27 fold {fold}: component={json.loads(details.read_text())["component_score"]:.9f}',flush=True)
    del model,x,base_real
    gc.collect()


def prediction_checkpoint(directory, arm, fold, fingerprint, held):
    base = baseline.prediction_checkpoint(None,'depth_mix',fold,None,held)
    if arm == 'model26':
        return base
    saved = read_checkpoint(directory/f'matched_context_fold{fold}.npz',
        dict(protocol_sha256=fingerprint,fold=fold,arm='matched_context'))
    if saved is None:
        raise ValueError('Incomplete Model27 fit')
    validate_predictions(saved,held)
    return candidate_probability(base,saved['prediction'])


def run():
    layout=load_development_layout(ROOT)
    pool=load_real_pool(layout)
    directory,protocol,fingerprint=initialize(layout,pool)
    print(f'Model27 run: {directory}',flush=True)
    real=read_cache(PARENT,'real_features',dict(protocol_sha256=PARENT_SHA,
        rows=layout.row_count,names=protocol['contract']['union_names'],
        tasks=int(np.sum(layout.inner_series_fold>=0))))
    context=contexts(pool)
    context_matrix=np.zeros((len(layout.inner_series_fold),len(CONTEXT_NAMES)),dtype=np.float32)
    for position,value in context.items(): context_matrix[position]=value
    for plan in protocol['plans']:
        fold=plan['fold']; meta=dict(protocol_sha256=fingerprint,fold=fold,arm='matched_context')
        held=np.flatnonzero(layout.row_fold==fold)
        if read_checkpoint(directory/f'matched_context_fold{fold}.npz',meta) is not None:
            continue
        tasks,sy,st,scontext=synthetic_tasks(pool,context,plan,layout)
        sx=build_cache(directory,f'matched_fold{fold}_features',tasks,len(sy),
            protocol['contract']['names'][fold],fingerprint)
        fit_one(directory,protocol,fingerprint,layout,real,context_matrix,plan,(sx,sy,st,scontext))
        del sx,tasks,sy,st,scontext
        gc.collect()
    from models.model27.report import report
    report()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('run','report'))
    args=parser.parse_args(); (HERE/'runs').mkdir(parents=True,exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='run': run()
        else:
            from models.model27.report import report
            report()


if __name__=='__main__': main()
