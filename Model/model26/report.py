"""Evaluate fixed equal-weight probability mixtures in the full ensemble."""
import json

import numpy as np
from scipy.special import expit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.run_model18 import read_checkpoint, commit_checkpoint
from models.model18.report import simultaneous_bounds, false_alarms
from models.model20.report import ensemble_margin
from models.model26.run_model26 import (ROOT, HERE, PARENT, ARMS, CONFIG,
    frozen_study, prediction_checkpoint)
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


def contrasts(means):
    values = np.asarray(means)
    return values[..., 1:]-values[..., :1]


def passes(result):
    return bool(result['mean_gain'] >= CONFIG['minimum_gain']
        and result['fold_wins'] >= CONFIG['minimum_fold_wins']
        and result['simultaneous_lower_95'] > 0)


def report():
    directory, protocol, protocol_sha = frozen_study()
    layout = load_development_layout(ROOT)
    predictions, component_scores = [], []
    for arm in ARMS:
        probability = np.empty(layout.row_count, dtype=np.float32)
        for fold in range(5):
            held = np.flatnonzero(layout.row_fold == fold)
            probability[held] = prediction_checkpoint(directory, arm, fold, protocol_sha, held)
        component_scores.append([tsauc(layout.target[layout.row_fold == f], probability[layout.row_fold == f],
            layout.time_online[layout.row_fold == f]) for f in range(5)])
        predictions.append(ensemble_margin(probability, layout))
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == f], p[layout.row_fold == f],
        layout.time_online[layout.row_fold == f]) for f in range(5)] for p in predictions])
    expected = [.6425650183808724,.6078580433856594,.6423542319802311,.6210564868818571,.6466398487375092]
    np.testing.assert_allclose(scores[0], expected, rtol=0, atol=1e-12,
        err_msg='Reconstructed Model23 baseline score differs')
    preliminary = dict(ensemble_scores=dict(zip(ARMS, scores.tolist())),
        component_scores=dict(zip(ARMS, component_scores)), confidence_checks_complete=False)
    write_json(directory/'preliminary_scores.json', preliminary)
    for arm, fold_scores in zip(ARMS, scores):
        print(f'{arm}: full ensemble={fold_scores.mean():.9f}, folds={fold_scores.tolist()}', flush=True)
    meta = dict(protocol_sha256=protocol_sha, predictions={str(p.name): sha256(p) for p in sorted(directory.glob('aligned_*_fold*.npz'))})
    checkpoint = directory/'bootstrap.npz'
    saved = read_checkpoint(checkpoint, meta)
    boot = [] if saved is None else saved['means'].tolist()
    count = CONFIG['bootstrap_replicates']
    if len(boot) > count or (boot and np.asarray(boot).shape != (len(boot), len(ARMS))):
        raise ValueError('Invalid Model26 bootstrap checkpoint')
    if len(boot) < count:
        scorers, series_folds = [], []
        for fold in range(5):
            rows = layout.row_fold == fold
            series_folds.append(np.flatnonzero(layout.inner_series_fold == fold))
            scorers.append([WeightedTSAUC(layout.target[rows], p[rows], layout.time_online[rows],
                layout.series_position[rows]) for p in predictions])
            np.testing.assert_allclose([s.score(np.ones(len(layout.inner_series_fold))) for s in scorers[-1]],
                scores[:, fold], rtol=0, atol=1e-12)
        rng = np.random.default_rng(CONFIG['bootstrap_seed'])
        completed = len(boot)
        for repeat in range(count):
            local = []
            for positions, fold_scorers in zip(series_folds, scorers):
                multiplicity = rng.multinomial(len(positions), np.full(len(positions), 1/len(positions)))
                if repeat < completed:
                    continue
                weights = np.zeros(len(layout.inner_series_fold))
                weights[positions] = multiplicity
                local.append([s.score(weights) for s in fold_scorers])
            if repeat < completed:
                continue
            boot.append(np.mean(local, axis=0).tolist())
            if (repeat+1) % 25 == 0 or repeat+1 == count:
                commit_checkpoint(checkpoint, dict(means=np.asarray(boot)), meta)
                print(f'Model26 whole-series bootstrap: {repeat+1}/{count}', flush=True)
    lower, critical = simultaneous_bounds(contrasts(scores.mean(axis=1)), contrasts(np.asarray(boot)), CONFIG['confidence'])
    results = {}
    for i, arm in enumerate(ARMS):
        result = describe(scores[i], scores[0])
        result['component'] = describe(component_scores[i], component_scores[0])
        result['no_break_false_alarms'] = false_alarms(expit(predictions[i].astype(np.float64)), layout, [.5, .8])
        if i:
            result['simultaneous_lower_95'] = float(lower[i-1])
            result['passes'] = passes(result)
        results[arm] = result
    passing = [arm for arm in ARMS[1:] if results[arm]['passes']]
    passing.sort(key=lambda arm: -results[arm]['mean_gain'])
    summary = dict(model=26, results=results, selected_candidate=passing[0] if passing else None,
        bootstrap_replicates=len(boot), simultaneous_critical_value=float(critical),
        interpretation=CONFIG['interpretation'],
        deployment='Model23 remains unchanged. No packaging, upload or cloud run performed.')
    write_json(directory/'summary.json', summary)
    lines = ['# Model26: fixed XGBoost probability mixtures', '',
        '| Mixture | Component CV | Full ensemble CV | Gain vs Model23 | Fold wins | Pass |',
        '|---|---:|---:|---:|---:|---|']
    for arm, result in results.items():
        lines.append(f"| {arm} | {result['component']['fold_mean']:.6f} | {result['fold_mean']:.6f} | "
            f"{result['mean_gain']:+.6f} | {result['fold_wins']}/5 | {result.get('passes', 'baseline')} |")
    lines += ['', 'Pass: gain >=0.0003, >=4/5 fold wins, simultaneous 95% lower bound >0 versus Model23.',
        f"Selected: {summary['selected_candidate'] or 'none; retain Model23'}.", '', CONFIG['interpretation'], '',
        'These are local development scores, not new cloud scores.', '', summary['deployment'], '']
    result_text = '\n'.join(lines)
    (directory/'RESULTS.md').write_text(result_text)
    (HERE/'RESULTS.md').write_text(result_text)
    print(result_text, flush=True)
    return summary

