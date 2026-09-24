"""Evaluate the retrained backbone both alone and inside frozen Model 16."""
import json
from pathlib import Path

import numpy as np
from scipy.special import logit, expit

from models.model10.sources import load_development_layout, _logistic
from models.model10.causal_transforms import TransformSpec, apply_transform_grid
from models.model11.pipeline import sha256, write_json, describe
from models.model18.run_model18 import read_checkpoint, commit_checkpoint, validate_predictions, digest
from models.model18.report import simultaneous_bounds, false_alarms
from models.model20.run_model20 import ROOT, HERE, SOURCE, STACK, COMPONENT, ARMS, CONFIG
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

# The completed run froze this source before encountering a NumPy-bool JSON
# serialization error. Permit only that exact old report hash, and record the
# repair hash in the output. Training, predictions and bootstrap stay frozen.
ORIGINAL_REPORT_SHA256 = 'fca8222899a3d4bb6aa2c581029cf9fc8a6ba54f77f6596adbfc2b42c1aeb520'


def primitive_logit(probability):
    bounded = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1-1e-6)
    return np.log(bounded)-np.log1p(-bounded)


def replace_stack_column(original_margin, original_probability, replacement, coefficient):
    delta = primitive_logit(replacement).astype(np.float64)-primitive_logit(original_probability).astype(np.float64)
    return (np.asarray(original_margin, dtype=np.float64)+coefficient*delta).astype(np.float32)


def ensemble_margin(probability, layout):
    old = np.load(SOURCE/'raw_oof.f32.npy')
    margin = np.load(STACK/'development_oof.margin.f32.npy')
    updated = np.empty_like(margin)
    for f in range(5):
        with np.load(STACK/f'pairwise_cv/fold_{f}.coefficients.npz') as coefficients:
            names = coefficients['feature_names'].tolist()
            weight = float(coefficients['raw_logit_coefficient'][names.index(COMPONENT)])
        rows = layout.row_fold == f
        updated[rows] = replace_stack_column(margin[rows], old[rows], probability[rows], weight)
    spec = TransformSpec(kind='rolling_max_premium', window=128, correction_weight=.1)
    _, corrected = next(apply_transform_grid(_logistic(updated).astype(np.float32),
        layout.time_online, layout.series_position, [spec]))
    compact = np.load(ROOT/'models/model11/runs/a982309521d5/oof.npy')
    return (.9*corrected+.1*logit(np.clip(compact, 1e-6, 1-1e-6))).astype(np.float32)


def contrasts(means):
    # Scores order: incumbent ensemble, fresh/raw/aligned ensemble, fresh/raw/aligned component.
    return np.asarray([means[2]-means[1], means[3]-means[1], means[2]-means[0], means[3]-means[0]])


def report(directory=None):
    latest = json.loads((HERE/'runs/latest.json').read_text())
    directory = ROOT/latest['directory'] if directory is None else Path(directory)
    protocol_sha = sha256(directory/'protocol.json')
    if protocol_sha != latest['protocol_sha256']:
        raise ValueError('Frozen Model 20 protocol checksum mismatch')
    protocol = json.loads((directory/'protocol.json').read_text())
    contract = protocol['contract']
    if contract['config'] != CONFIG or digest(protocol['plans']) != protocol['plans_sha256']:
        raise ValueError('Frozen study changed')
    repair = None
    for relative, expected in contract['hashes'].items():
        if sha256(ROOT/relative) != expected:
            if relative == 'models/model20/report.py' and expected == ORIGINAL_REPORT_SHA256:
                repair = dict(reason='Convert NumPy promotion flag to JSON-compatible Python bool; no scoring changes',
                    original_report_sha256=expected, repaired_report_sha256=sha256(ROOT/relative))
                continue
            raise ValueError(f'Changed source or input: {relative}')
    layout = load_development_layout(ROOT)
    components = []
    for arm in ARMS:
        p = np.empty(layout.row_count, dtype=np.float32)
        for fold in range(5):
            held = np.flatnonzero(layout.row_fold == fold)
            saved = read_checkpoint(directory/f'{arm}_fold{fold}.npz',
                dict(protocol_sha256=protocol_sha, fold=fold, arm=arm))
            if saved is None:
                raise RuntimeError(f'Incomplete study: {arm} fold {fold}')
            validate_predictions(saved, held)
            p[held] = saved['prediction']
        components.append(p)
    incumbent = np.load(ROOT/'models/model16/runs/ee6c40a49bf7/oof.margin.npy')
    reconstructed = ensemble_margin(np.load(SOURCE/'raw_oof.f32.npy'), layout)
    np.testing.assert_array_equal(reconstructed, incumbent,
        err_msg='Original stack-column identity does not reproduce Model 16')
    predictions = [incumbent, *[ensemble_margin(p, layout) for p in components], *components]
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == f], p[layout.row_fold == f],
        layout.time_online[layout.row_fold == f]) for f in range(5)] for p in predictions])
    names = ['incumbent', *[f'ensemble_{a}' for a in ARMS], *[f'component_{a}' for a in ARMS]]
    for name, s in zip(names, scores):
        print(f'{name}: local TSAUC={s.mean():.9f}, folds={s.tolist()}', flush=True)
    write_json(directory/'preliminary_scores.json', dict(scores=dict(zip(names, scores.tolist())), confidence_checks_complete=False))
    checkpoint = directory/'bootstrap.npz'
    meta = dict(protocol_sha256=protocol_sha, predictions={f'{a}_{f}':sha256(directory/f'{a}_fold{f}.npz')
        for a in ARMS for f in range(5)})
    saved = read_checkpoint(checkpoint, meta)
    boot = [] if saved is None else saved['means'].tolist()
    count = CONFIG['bootstrap_replicates']
    if len(boot) > count or (boot and np.asarray(boot).shape != (len(boot), 7)):
        raise ValueError('Invalid bootstrap checkpoint')
    if len(boot) < count:
        scorers, series_folds = [], []
        for fold in range(5):
            rows = layout.row_fold == fold
            series_folds.append(np.flatnonzero(layout.inner_series_fold == fold))
            scorers.append([WeightedTSAUC(layout.target[rows], p[rows], layout.time_online[rows],
                layout.series_position[rows]) for p in predictions])
            np.testing.assert_allclose([s.score(np.ones(len(layout.inner_series_fold))) for s in scorers[-1]],
                scores[:, fold], atol=1e-12, rtol=0)
        rng = np.random.default_rng(CONFIG['bootstrap_seed'])
        completed = len(boot)
        for repeat in range(count):
            local = []
            for positions, scorers_fold in zip(series_folds, scorers):
                multiplicity = rng.multinomial(len(positions), np.full(len(positions), 1/len(positions)))
                if repeat < completed:
                    continue
                weights = np.zeros(len(layout.inner_series_fold))
                weights[positions] = multiplicity
                local.append([s.score(weights) for s in scorers_fold])
            if repeat < completed:
                continue
            boot.append(np.mean(local, axis=0).tolist())
            if (repeat+1) % 25 == 0:
                commit_checkpoint(checkpoint, dict(means=np.asarray(boot)), meta)
                print(f'Backbone whole-series bootstrap: {repeat+1}/{count}', flush=True)
    lower, critical = simultaneous_bounds(contrasts(scores.mean(axis=1)),
        np.asarray([contrasts(m) for m in boot]), CONFIG['confidence'])
    results = {}
    for i, arm in enumerate(ARMS):
        result = dict(component=describe(scores[i+4], scores[4]),
            ensemble_vs_fresh_control=describe(scores[i+1], scores[1]),
            ensemble_vs_incumbent=describe(scores[i+1], scores[0]),
            no_break_false_alarms=false_alarms(expit(predictions[i+1].astype(np.float64)), layout, [.5, .8]))
        if i:
            result['lower_vs_fresh_control'] = float(lower[i-1])
            result['lower_vs_incumbent'] = float(lower[i+1])
            result['passes'] = bool(all(result[key]['mean_gain'] >= CONFIG['minimum_gain']
                and result[key]['fold_wins'] >= CONFIG['minimum_fold_wins']
                for key in ('ensemble_vs_fresh_control', 'ensemble_vs_incumbent')) and min(lower[i-1], lower[i+1]) > 0)
        results[arm] = result
    passing = [a for a in ARMS[1:] if results[a]['passes']]
    passing.sort(key=lambda a: (-results[a]['ensemble_vs_incumbent']['mean_gain'], int(a.split('_')[1])))
    summary = dict(model=20, results=results, selected_candidate=passing[0] if passing else None,
        incumbent_scores=scores[0].tolist(), historic_component_scores=[r['tsroc'] for r in contract['recipes']],
        simultaneous_critical_value=critical, bootstrap_replicates=len(boot),
        interpretation=CONFIG['interpretation'], report_repair=repair,
        deployment='Model 16 unchanged; no submission performed by the research runner')
    write_json(directory/'summary.json', summary)
    lines = ['# Model 20: retrain a deployed Model 16 backbone', '',
        'Core XGBoost, original fold-specific 256 features, depth3, 800 rounds, 400k real fitting rows.',
        'The raw10/aligned25 arms add up to400k synthetic rows; original weights are conserved by (step,target).', '',
        '| Arm | Component TSAUC | Full ensemble TSAUC | Gain vs incumbent | Gain vs fresh control | Pass |',
        '|---|---:|---:|---:|---:|---|', f'| Original Model 16 | — | {scores[0].mean():.6f} | — | — | incumbent |']
    for arm, result in results.items():
        lines.append(f"| {arm} | {result['component']['fold_mean']:.6f} | {result['ensemble_vs_incumbent']['fold_mean']:.6f} | "
            f"{result['ensemble_vs_incumbent']['mean_gain']:+.6f} | {result['ensemble_vs_fresh_control']['mean_gain']:+.6f} | {result.get('passes', 'control')} |")
    lines += ['', 'Pass requires full-ensemble gain >=0.0003, >=4/5 fold wins, and simultaneous lower bound >0 against BOTH fresh control and original Model16.',
        f"Selected: {summary['selected_candidate'] or 'none; keep original Model 16'}.", '', CONFIG['interpretation'], '',
        'These are local development scores, not new cloud scores. Component refits may differ from historical fits due to dependency versions; the fresh paired control isolates augmentation.', '',
        summary['deployment'], '']
    text = '\n'.join(lines)
    (directory/'RESULTS.md').write_text(text)
    (HERE/'RESULTS.md').write_text(text)
    print(text, flush=True)
    return summary
