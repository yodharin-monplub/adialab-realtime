"""Strong deployed-backbone retraining, isolated from all incumbent artifacts."""
from pathlib import Path
import argparse
import fcntl
import gc
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
from models.model18.augmentation import select_pairs, assemble
from models.model18.run_model18 import digest, commit_checkpoint, read_checkpoint, validate_predictions
from models.model20.features import BUNDLE, build_cache, load_real_pool, selected_features
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'models/model6/runs/tsroc0p64_909b8f5c6c/final_cv/core/xgboost__fisher_256__xgb_depth3'
STACK = ROOT/'models/model9/runs/fulldata_819a001db6/pairwise_stack'
COMPONENT = 'core:feature:xgboost__fisher_256__xgb_depth3'
NATIVE = BUNDLE/'resources/feature_models/core__feature__xgboost__fisher_256__xgb_depth3'
ARMS = ('control', 'raw_10', 'aligned_25')
CONFIG = dict(version=1, arms=list(ARMS), seed=20260920, workers=4, threads=4,
    fit_rows=400_000, synthetic_rows=400_000, pairs=1000, max_b_reuse=5,
    augmentation='Raw10 and aligned25: prior standalone and full-ensemble leaders; frozen follow-up',
    original_sampling_fraction=.2, original_sampling_seed=99173,
    bootstrap_replicates=1000, bootstrap_seed=200906, minimum_gain=.0003,
    minimum_fold_wins=4, confidence=.95,
    interpretation='Reused development experiment, not fresh confirmation. Existing feature rankings '
        'and stack weights remain fixed and development-informed. Conditional bootstrap excludes '
        'upstream refitting/selection uncertainty. No reserved holdout or test-label evaluation.')


def sampled_mask(row_id, steps):
    value = (np.asarray(row_id, dtype=np.uint64)*np.uint64(0x9E3779B185EBCA87)
             + np.asarray(steps, dtype=np.uint64)*np.uint64(0xC2B2AE3D27D4EB4F)+np.uint64(99173))
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value % np.uint64(1000003) < int(.2*1000003)


def fitting_rows(layout, fold, seed):
    fitting = (layout.row_fold != fold) & (layout.row_fold != (fold+1) % 5)
    sample = sampled_mask(layout.series_ids[layout.series_position], layout.time_online)
    eligible = np.flatnonzero(fitting & sample)
    selected = eligible if len(eligible) <= CONFIG['fit_rows'] else np.sort(
        np.random.default_rng(seed+1).choice(eligible, CONFIG['fit_rows'], replace=False))
    return selected, fitting


def reference_weights(sample_y, sample_t, reference_y, reference_t):
    size = int(max(sample_t.max(initial=0), reference_t.max(initial=0)))+1
    counts = np.bincount(reference_t, minlength=size)
    positive = np.bincount(reference_t, weights=reference_y, minlength=size)
    negative = counts-positive
    weights = np.where(sample_y == 1, negative[sample_t], positive[sample_t]).astype(np.float64)
    useful = weights > 0
    if useful.any():
        weights[useful] /= weights[useful].mean()
    return weights.astype(np.float32)


def splice_weights(original, real_y, real_t, sy, st, share):
    rc, sc = 2*real_t.astype(np.int64)+real_y, 2*st.astype(np.int64)+sy
    size = int(max(rc.max(initial=0), sc.max(initial=0)))+1
    mass = np.bincount(rc, weights=original, minlength=size)
    count = np.bincount(sc, minlength=size)
    rw = (original*np.where(count[rc] > 0, 1-share, 1)).astype(np.float32)
    sw = np.divide(share*mass, count, out=np.zeros(size), where=count > 0)[sc].astype(np.float32)
    after = np.bincount(rc, weights=rw, minlength=size)+np.bincount(sc, weights=sw, minlength=size)
    np.testing.assert_allclose(after, mass, rtol=2e-7, atol=1e-6)
    return rw, sw, dict(actual_share=float(sw.sum(dtype=np.float64)/original.sum(dtype=np.float64)),
        original_total=float(mass.sum()), final_total=float(after.sum()), max_cell_error=float(np.max(np.abs(after-mass))))


def source_files():
    files = [*HERE.glob('*.py'), *ROOT.glob('models/model18/*.py'),
        ROOT/'models/model10/sources.py', ROOT/'models/model10/causal_transforms.py',
        ROOT/'models/model11/pipeline.py', ROOT/'models/report_next_models.py', ROOT/'tsauc.py',
        ROOT/'train/X_train.parquet', ROOT/'train/y_train_index.parquet',
        ROOT/'models/model5/artifacts/split_assignments.npz', ROOT/'models/artifacts/core_features.metadata.npz',
        SOURCE/'raw_oof.f32.npy', STACK/'development_oof.margin.f32.npy',
        ROOT/'models/model11/runs/a982309521d5/oof.npy',
        ROOT/'models/model16/runs/ee6c40a49bf7/oof.margin.npy', BUNDLE/'feature_pipeline.py', BUNDLE/'competition_data.py']
    files += list((BUNDLE/'detector').rglob('*.py'))
    for f in range(5):
        files += [SOURCE/f'fold_{f}.json', SOURCE/f'fold_{f}.columns.npz', NATIVE/f'fold_{f}.ubj',
                  STACK/f'pairwise_cv/fold_{f}.coefficients.npz']
    return sorted(set(files))


def initialize(layout, pool):
    recipes = [json.loads((SOURCE/f'fold_{f}.json').read_text()) for f in range(5)]
    names = [np.load(SOURCE/f'fold_{f}.columns.npz')['names'].tolist() for f in range(5)]
    contract = dict(config=CONFIG, recipes=recipes, names=names, union_names=sorted(set(sum(names, []))),
        hashes={str(p.relative_to(ROOT)): sha256(p) for p in source_files()},
        dependencies={**{p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'pyarrow', 'scikit-learn')},
                      'xgboost': xgb.__version__, 'xgboost_distribution': importlib.metadata.version('xgboost-cpu')},
        python=sys.version)
    directory = HERE/'runs'/digest(contract)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'protocol.json'
    if path.exists():
        protocol = json.loads(path.read_text())
        if protocol['contract'] != contract or digest(protocol['plans']) != protocol['plans_sha256']:
            raise ValueError('Frozen protocol mismatch')
    else:
        plans = []
        for fold, recipe in enumerate(recipes):
            fit, _ = fitting_rows(layout, fold, recipe['seed'])
            donors = {p: values for p, values in pool.items() if layout.inner_series_fold[p] not in (fold, (fold+1) % 5)}
            pairs, stats = select_pairs(donors, layout.tau_index, layout.inner_series_fold,
                fold, CONFIG['seed']+fold, count=CONFIG['pairs'], reuse=CONFIG['max_b_reuse'])
            if not pairs:
                raise ValueError('No fitting-fold donors')
            total = sum(pair['length'] for pair in pairs)
            selected = np.sort(np.random.default_rng(np.random.SeedSequence([CONFIG['seed']+fold, 1])).choice(
                total, min(total, CONFIG['synthetic_rows']), replace=False))
            plans.append(dict(fold=fold, fit_rows=fit.tolist(), pairs=pairs, stats=stats,
                              selected=selected.tolist(), synthetic_total=total))
            print(f'Frozen fold {fold}: {len(fit):,} real rows; {stats}', flush=True)
        protocol = dict(contract=contract, plans=plans, plans_sha256=digest(plans))
        write_json(path, protocol)
    protocol_sha = sha256(path)
    write_json(HERE/'runs/latest.json', dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=protocol_sha))
    return directory, protocol, protocol_sha


def native_parity(directory, layout, pool, names, protocol_sha):
    path = directory/'native_parity.json'
    if path.exists():
        record = json.loads(path.read_text())
        if record['protocol_sha256'] != protocol_sha or max(record['errors']) > 1e-6:
            raise ValueError('Invalid native parity checkpoint')
        return
    original = np.load(SOURCE/'raw_oof.f32.npy')
    errors = []
    for fold in range(5):
        model = xgb.XGBClassifier()
        model.load_model(NATIVE/f'fold_{fold}.ubj')
        model.set_params(n_jobs=1)
        for p in np.flatnonzero(layout.inner_series_fold == fold)[:3]:
            h, o = pool[p]
            x = selected_features(h, o, names[fold])
            prediction = model.predict_proba(x)[:, 1]
            reference = original[layout.series_position == p]
            error = float(np.max(np.abs(prediction-reference)))
            if error > 1e-6:
                raise ValueError(f'Rebuilt deployed features fail prediction parity: {error}')
            errors.append(error)
    write_json(path, dict(protocol_sha256=protocol_sha, errors=errors, tested_series=len(errors)))
    print(f'Deployed XGBoost prediction parity: maximum error={max(errors)} on {len(errors)} series', flush=True)


def synthetic_tasks(pool, plan, layout, aligned):
    tasks, targets, times = [], [], []
    selected = np.asarray(plan['selected'], dtype=np.int64)
    offset = 0
    for pair in plan['pairs']:
        for key in ('a', 'b'):
            if layout.inner_series_fold[pair[key]] in (-1, plan['fold'], (plan['fold']+1) % 5):
                raise ValueError('Synthetic donor leaks scoring or reserved role')
        h, o, y, t = assemble(pool, pair, aligned)
        left, right = np.searchsorted(selected, [offset, offset+len(o)])
        keep = selected[left:right]-offset
        tasks.append((h, o, keep))
        targets.append(y[keep])
        times.append(t[keep])
        offset += len(o)
    return tasks, np.concatenate(targets), np.concatenate(times)


def train_one(directory, protocol, protocol_sha, layout, real_x, plan, arm, synthetic=None):
    fold = plan['fold']
    path = directory/f'{arm}_fold{fold}.npz'
    meta = dict(protocol_sha256=protocol_sha, fold=fold, arm=arm)
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
    x = real_x[fit][:, columns]
    accounting = dict(actual_share=0.)
    if synthetic is not None:
        sx, sy, st = synthetic
        rw, sw, accounting = splice_weights(weights, y, t, sy, st, int(arm.split('_')[1])/100)
        x, y, weights = np.concatenate((x, sx)), np.r_[y, sy], np.r_[rw, sw]
    started = time.monotonic()
    params = {k: v for k, v in recipe['hyperparameters'].items() if k != 'name'}
    model = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', n_estimators=recipe['rounds'],
        eval_metric='logloss', random_state=recipe['seed'], n_jobs=4, device='cpu', verbosity=0, **params)
    print(f'Fitting deployed XGBoost {arm}, fold {fold}: {len(y):,} rows, {len(columns)} features, {recipe["rounds"]} rounds', flush=True)
    model.fit(x, y, sample_weight=weights, verbose=False)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        prediction[left:left+50_000] = model.predict_proba(real_x[held[left:left+50_000]][:, columns])[:, 1]
    model_path = directory/f'{arm}_fold{fold}.ubj'
    temporary = directory/f'{arm}_fold{fold}.partial.ubj'
    model.save_model(temporary)
    temporary.replace(model_path)
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    details_path = directory/f'{arm}_fold{fold}.fit.json'
    write_json(details_path, dict(score=score, historical_component_score=recipe['tsroc'],
        real_rows=len(fit), synthetic_rows=0 if synthetic is None else len(synthetic[1]), weights=accounting,
        elapsed_seconds=time.monotonic()-started))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta, (model_path, details_path))
    print(f'Finished {arm}, fold {fold}: component TSAUC={score:.9f}, {time.monotonic()-started:.1f}s', flush=True)
    del model, x
    gc.collect()


def run():
    layout = load_development_layout(ROOT)
    pool = load_real_pool(layout)
    directory, protocol, protocol_sha = initialize(layout, pool)
    native_parity(directory, layout, pool, protocol['contract']['names'], protocol_sha)
    tasks = [(*pool[p], None) for p in np.flatnonzero(layout.inner_series_fold >= 0)]
    real_x = build_cache(directory, 'real_features', tasks, layout.row_count,
        protocol['contract']['union_names'], protocol_sha)
    del tasks
    for plan in protocol['plans']:
        train_one(directory, protocol, protocol_sha, layout, real_x, plan, 'control')
    for plan in protocol['plans']:
        fold = plan['fold']
        for arm in ARMS[1:]:
            meta = dict(protocol_sha256=protocol_sha, fold=fold, arm=arm)
            if read_checkpoint(directory/f'{arm}_fold{fold}.npz', meta) is not None:
                continue
            tasks, sy, st = synthetic_tasks(pool, plan, layout, arm.startswith('aligned'))
            sx = build_cache(directory, f'{arm}_fold{fold}_features', tasks, len(sy),
                protocol['contract']['names'][fold], protocol_sha)
            train_one(directory, protocol, protocol_sha, layout, real_x, plan, arm, (sx, sy, st))
            del sx, tasks, sy, st
            gc.collect()
    from models.model20.report import report
    report(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report'))
    args = parser.parse_args()
    (HERE/'runs').mkdir(parents=True, exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Model20 is already running; inspect its checkpoints for status')
        if args.command == 'run':
            run()
        else:
            from models.model20.report import report
            report()
