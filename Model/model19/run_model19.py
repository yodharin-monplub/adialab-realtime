"""Evaluate augmentation inside the existing Model 16 ensemble, CPU only."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys

import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout, load_model9_pairwise_stack
from models.model10.causal_transforms import TransformSpec, apply_transform_grid
from models.model11.pipeline import sha256, write_json, describe
from models.model18.run_model18 import (read_checkpoint, commit_checkpoint, checkpoint_meta,
    validate_predictions, digest)
from models.model18.report import simultaneous_bounds, false_alarms
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
MODEL16 = ROOT / 'models/model16/runs/ee6c40a49bf7'
MODEL18 = ROOT / 'models/model18/runs/ae6d2a6154348e82'
ARMS = ('control', 'raw_10', 'aligned_10', 'raw_25', 'aligned_25')
CONFIG = dict(version=1, model=19, baseline='Model 16 (cloud score approximately 0.622)',
    arms=list(ARMS), component_weight=.1, backbone_weight=.9, rolling_window=128,
    rolling_weight=.1, bootstrap_replicates=1000, bootstrap_seed=190906,
    minimum_gain=.0003, minimum_fold_wins=4, confidence=.95,
    false_alarm_thresholds=[.5, .8], control_tolerance=1e-7,
    protocol='Frozen Model 16 backbone and weights; swap only its compact component',
    interpretation='Exploratory continuation on reused development folds, not fresh confirmation. '
        'Model 18 augmentation and Model 16 upstream selection are already development-informed. '
        'Conditional paired bootstrap excludes upstream training/selection uncertainty.')


def combine(backbone, compact_probability):
    backbone = np.asarray(backbone)
    probability = np.asarray(compact_probability)
    if (backbone.shape != probability.shape or not np.isfinite(backbone).all()
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Invalid backbone or compact predictions')
    return (.9 * backbone + .1 * logit(np.clip(probability, 1e-6, 1-1e-6))).astype(np.float32)


def passes(result):
    return (result['mean_gain'] >= CONFIG['minimum_gain']
        and result['fold_wins'] >= CONFIG['minimum_fold_wins']
        and result['simultaneous_lower_95'] > 0)


def initialize():
    source_contract = json.loads((MODEL18/'protocol.json').read_text())['contract']
    for relative, expected in (
        ('train/y_train_index.parquet', source_contract['features']['target_index_sha256']),
        ('models/model5/artifacts/split_assignments.npz', source_contract['features']['split_sha256']),
        ('models/artifacts/core_features.metadata.npz', source_contract['file_hashes']['models/artifacts/core_features.metadata.npz'])):
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Augmentation source alignment changed: {relative}')
    paths = [*HERE.glob('*.py'), *ROOT.glob('models/model18/*.py'),
        ROOT/'models/model10/sources.py', ROOT/'models/model10/causal_transforms.py',
        ROOT/'models/report_next_models.py', ROOT/'tsauc.py',
        ROOT/'train/y_train_index.parquet', ROOT/'models/model5/artifacts/split_assignments.npz',
        ROOT/'models/artifacts/core_features.metadata.npz',
        MODEL16/'oof.margin.npy', MODEL16/'summary.json', MODEL16/'protocol.json',
        MODEL18/'protocol.json', MODEL18/'summary.json',
        ROOT/'models/model9/runs/fulldata_819a001db6/pairwise_stack/development_oof.margin.f32.npy']
    paths += [MODEL18/f'{arm}_fold{fold}.{extension}' for arm in ARMS for fold in range(5)
              for extension in ('npz', 'json', 'fit.json')]
    contract = dict(config=CONFIG, hashes={str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)},
        dependencies={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas')},
        expected_control_scores=json.loads((MODEL16/'summary.json').read_text())['comparison_to_model6']['fold_scores'])
    directory = HERE/'runs'/digest(contract)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Frozen Model 19 protocol differs')
    else:
        write_json(path, contract)
    write_json(HERE/'runs/latest.json', dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=sha256(path)))
    return directory, contract, sha256(path)


def prepare_predictions(directory, contract, protocol_sha):
    layout = load_development_layout(ROOT)
    source_protocol_sha = sha256(MODEL18/'protocol.json')
    original = np.load(MODEL16/'oof.margin.npy')
    if original.shape != (layout.row_count,) or not np.isfinite(original).all():
        raise ValueError('Invalid Model 16 baseline')
    source = load_model9_pairwise_stack(ROOT, layout=layout)
    spec = TransformSpec(kind='rolling_max_premium', window=128, correction_weight=.1)
    _, backbone = next(apply_transform_grid(source.probability, layout.time_online, layout.series_position, [spec]))
    # Existing Model 18 fits are training-fold-only and need not be retrained.
    for arm in ARMS:
        prediction = np.empty(layout.row_count, dtype=np.float32)
        for fold in range(5):
            held = np.flatnonzero(layout.row_fold == fold)
            saved = read_checkpoint(MODEL18/f'{arm}_fold{fold}.npz', checkpoint_meta(source_protocol_sha, fold, arm))
            if saved is None:
                raise ValueError('Source augmentation study is incomplete')
            validate_predictions(saved, held)
            prediction[held] = combine(backbone[held], saved['prediction'])
        if arm == 'control':
            np.testing.assert_array_equal(prediction, original,
                err_msg='Reconstructed full Model 16 control differs from deployed-recipe OOF')
        path = directory/f'{arm}.npz'
        meta = dict(protocol_sha256=protocol_sha, arm=arm)
        saved = read_checkpoint(path, meta)
        if saved is None:
            commit_checkpoint(path, dict(margin=prediction), meta)
        else:
            np.testing.assert_array_equal(saved['margin'], prediction)
        scores = [tsauc(layout.target[layout.row_fold == f], prediction[layout.row_fold == f],
                        layout.time_online[layout.row_fold == f]) for f in range(5)]
        print(f'Model 16 + {arm}: TSAUC={np.mean(scores):.9f}; folds={scores}', flush=True)


def report(directory=None):
    latest = json.loads((HERE/'runs/latest.json').read_text())
    directory = ROOT/latest['directory'] if directory is None else Path(directory)
    protocol_sha = sha256(directory/'protocol.json')
    if protocol_sha != latest['protocol_sha256']:
        raise ValueError('Protocol checksum mismatch')
    contract = json.loads((directory/'protocol.json').read_text())
    if contract['config'] != CONFIG:
        raise ValueError('Report settings differ from frozen protocol')
    for relative, checksum in contract['hashes'].items():
        if sha256(ROOT/relative) != checksum:
            raise ValueError(f'Changed source/input: {relative}')
    layout = load_development_layout(ROOT)
    predictions, scores = [], []
    for arm in ARMS:
        saved = read_checkpoint(directory/f'{arm}.npz', dict(protocol_sha256=protocol_sha, arm=arm))
        if saved is None:
            raise RuntimeError(f'Missing {arm}; run preparation first')
        p = saved['margin']
        if p.shape != (layout.row_count,) or not np.isfinite(p).all():
            raise ValueError('Invalid full-ensemble margins')
        predictions.append(p)
        scores.append([tsauc(layout.target[layout.row_fold == f], p[layout.row_fold == f],
                             layout.time_online[layout.row_fold == f]) for f in range(5)])
    scores = np.asarray(scores)
    np.testing.assert_allclose(scores[0], contract['expected_control_scores'], atol=CONFIG['control_tolerance'], rtol=0)
    observed = scores.mean(axis=1)[1:]-scores[0].mean()
    checkpoint = directory/'bootstrap.npz'
    meta = dict(protocol_sha256=protocol_sha,
                predictions={arm: sha256(directory/f'{arm}.npz') for arm in ARMS})
    saved = read_checkpoint(checkpoint, meta)
    differences = [] if saved is None else saved['differences'].tolist()
    count = CONFIG['bootstrap_replicates']
    if len(differences) > count or (differences and np.asarray(differences).shape != (len(differences), 4)):
        raise ValueError('Invalid bootstrap checkpoint')
    if len(differences) < count:
        scorers, fold_series = [], []
        for fold in range(5):
            held = layout.row_fold == fold
            fold_series.append(np.flatnonzero(layout.inner_series_fold == fold))
            scorers.append([WeightedTSAUC(layout.target[held], p[held], layout.time_online[held],
                                          layout.series_position[held]) for p in predictions])
            np.testing.assert_allclose([s.score(np.ones(len(layout.inner_series_fold))) for s in scorers[-1]],
                                       scores[:, fold], atol=1e-12, rtol=0)
        rng = np.random.default_rng(CONFIG['bootstrap_seed'])
        completed = len(differences)
        for repeat in range(count):
            local_scores = []
            for series, local in zip(fold_series, scorers):
                multiplicity = rng.multinomial(len(series), np.full(len(series), 1/len(series)))
                if repeat < completed:
                    continue
                weights = np.zeros(len(layout.inner_series_fold))
                weights[series] = multiplicity
                local_scores.append([s.score(weights) for s in local])
            if repeat < completed:
                continue
            means = np.mean(local_scores, axis=0)
            differences.append((means[1:]-means[0]).tolist())
            if (repeat+1) % 25 == 0:
                commit_checkpoint(checkpoint, dict(differences=np.asarray(differences)), meta)
                print(f'Full Model 16 paired bootstrap: {repeat+1}/{count}', flush=True)
    lower, critical = simultaneous_bounds(observed, differences, CONFIG['confidence'])
    results = {}
    for i, arm in enumerate(ARMS):
        result = describe(scores[i], scores[0])
        result['fold_gains'] = (scores[i]-scores[0]).tolist()
        result['no_break_false_alarms'] = false_alarms(expit(predictions[i].astype(np.float64)), layout, [.5, .8])
        if i:
            result['simultaneous_lower_95'] = float(lower[i-1])
            result['passes'] = passes(result)
        results[arm] = result
    passing = [a for a in ARMS[1:] if results[a]['passes']]
    passing.sort(key=lambda a: (-results[a]['mean_gain'], int(a.split('_')[1]), not a.startswith('raw')))
    summary = dict(model=19, results=results, selected_candidate=passing[0] if passing else None,
        bootstrap_replicates=len(differences), simultaneous_critical_value=critical,
        interpretation=CONFIG['interpretation'], reserved_holdout_reopened=False,
        competition_labels_loaded=False, deployment='Model 16 unchanged; nothing submitted')
    write_json(directory/'summary.json', summary)
    lines = ['# Model 19: continue from Model 16', '',
        'The approximately 0.622 cloud submission is the control. Values below are LOCAL development TSAUC, not cloud scores.', '',
        'Keep its 90% backbone and persistence unchanged; replace only its 10% compact component with splice-trained fits.', '',
        '| Replacement | Full ensemble TSAUC | Gain vs Model 16 | Fold wins | Simultaneous 95% lower gain |',
        '|---|---:|---:|---:|---:|']
    for arm, result in results.items():
        bound = '—' if arm == 'control' else f"{result['simultaneous_lower_95']:+.6f}"
        lines.append(f"| {arm} | {result['fold_mean']:.6f} | {result['mean_gain']:+.6f} | {result['fold_wins']}/5 | {bound} |")
    lines += ['', 'Pass requires gain >= 0.0003, at least 4/5 fold wins, and simultaneous lower bound > 0.',
        f"Selected: {summary['selected_candidate'] or 'none; keep original Model 16'}.", '', CONFIG['interpretation'], '',
        'Full fold gains and fixed-threshold false-alarm counts are saved in the run summary.', '', summary['deployment'], '']
    (directory/'RESULTS.md').write_text('\n'.join(lines))
    (HERE/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report'))
    args = parser.parse_args()
    if args.command == 'run':
        directory, contract, protocol_sha = initialize()
        prepare_predictions(directory, contract, protocol_sha)
        report(directory)
    else:
        report()
