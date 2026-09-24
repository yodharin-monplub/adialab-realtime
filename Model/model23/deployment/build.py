"""Build a separate, unsealed Submission23 bundle after local research gates."""
from pathlib import Path
import json
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import read_checkpoint
from models.model18.report import simultaneous_bounds
from models.model23.run_model23 import ARMS, CONFIG, PARENT, MODEL21, frozen_study

HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'submissions/submission20'
DESTINATION = ROOT/'submissions/submission23'
OUTPUT = ROOT/'models/model23/deployment_validation'
BASE_SHA = 'ed80e6b06bda7dadce216c30d61f699babb6c46f5d7dbcbc96fac1b96d1d7818'
COMPONENT = 'core__feature__xgboost__fisher_256__xgb_depth3'


def checked_source(root, name):
    path = root/name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe resource path')
    return path


def main():
    if DESTINATION.exists():
        raise FileExistsError('Submission23 already exists; refusing to overwrite')
    run, protocol, fingerprint = frozen_study()
    result = json.loads((run/'summary.json').read_text())
    if result['selected_candidate'] != 'mix_60' or result['bootstrap_replicates'] != 1000:
        raise ValueError('Expected complete passing mix_60 study')
    bootstrap = read_checkpoint(run/'bootstrap.npz',
        dict(protocol_sha256=fingerprint, predictions=protocol['contract']['prediction_hashes']))
    if bootstrap is None or bootstrap['means'].shape != (1000, 3):
        raise ValueError('Incomplete verified bootstrap')
    scores = np.asarray([result['results'][arm]['fold_scores'] for arm in ARMS])
    observed = scores[1:].mean(axis=1)-scores[0].mean()
    lower, _ = simultaneous_bounds(observed, bootstrap['means'][:, 1:]-bootstrap['means'][:, :1], .95)
    if not (observed[1] >= CONFIG['minimum_gain'] and np.sum(scores[2] > scores[0]) >= 4 and lower[1] > 0):
        raise ValueError('Candidate fails recomputed local gates')
    if sha256(SOURCE/'bundle_manifest.json') != BASE_SHA:
        raise ValueError('Source differs from uploaded Submission20 seal')
    sealed = json.loads((SOURCE/'bundle_manifest.json').read_text())
    for name, record in sealed['files'].items():
        if sha256(checked_source(SOURCE, name)) != record['sha256']:
            raise ValueError(f'Changed Submission20 file: {name}')
    parent = json.loads((PARENT/'protocol.json').read_text())
    source_manifest = json.loads((SOURCE/'resources/manifest.json').read_text())
    component = next(c for c in source_manifest['feature_components'] if c['name'] == COMPONENT)
    if len(component['voters']) != 5:
        raise ValueError('Expected five base voters')
    for fold, voter in enumerate(component['voters']):
        if voter['feature_names'] != parent['contract']['names'][fold]:
            raise ValueError('Fold feature names changed')
        if sha256(SOURCE/'resources'/voter['path']) != sha256(PARENT/f'aligned_25_fold{fold}.ubj'):
            raise ValueError('Base voter differs from Model20 research checkpoint')
    staging = Path(tempfile.mkdtemp(prefix='submission23-staging-', dir=ROOT/'submissions'))
    for name in sealed['files']:
        if name in {'main.py', 'solution20.py', 'readme.txt'}:
            continue
        target = staging/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked_source(SOURCE, name), target)
    additions = []
    for fold, voter in enumerate(component['voters']):
        name = f'feature_models/{COMPONENT}/paired_aligned60_fold_{fold}.ubj'
        shutil.copy2(MODEL21/f'aligned_60_fold{fold}.ubj', staging/'resources'/name)
        voter['paired_path'] = name
        source_manifest['file_sha256'][name] = sha256(staging/'resources'/name)
        additions.append(name)
    source_manifest.update(submission='submission_23', backbone_augmentation='mix_60',
        training_protocol_sha256=fingerprint, study_summary_sha256=sha256(run/'summary.json'),
        pairing_recipe='float32 mean of aligned25/aligned60 probabilities within fold, then original five-fold logit bag',
        validation_note='Exploratory OOF score0.632094726; five-pair deployment requires separate serving validation and cloud confirmation.')
    write_json(staging/'resources/manifest.json', source_manifest)
    for source, target in [('entry.py', 'main.py'), ('solution.py', 'solution23.py'),
                           ('pairing.py', 'pairing.py'), ('readme.txt', 'readme.txt')]:
        shutil.copy2(HERE/source, staging/target)
    staging.rename(DESTINATION)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT/'models/model20/cloud_run_settings.json').read_text())
    settings.pop('submissionId', None)
    settings['name'] = 'Submission23 — paired synthetic-weight XGBoost, pretrained inference only'
    if settings['forceFirstTrain'] is not False or settings['trainFrequency'] != 0:
        raise ValueError('Cloud configuration is not inference-only')
    write_json(OUTPUT/'cloud_run_settings.json', settings)
    write_json(OUTPUT/'build.json', dict(base_bundle_sha256=BASE_SHA, model23_run=str(run.relative_to(ROOT)),
        study_protocol_sha256=fingerprint, summary_sha256=sha256(run/'summary.json'),
        added_resource_files=additions, all_original_model_resources_unchanged=True,
        within_fold_probability_mixture=True, forceFirstTrain=False, infer_parallelism=12,
        sealed=False, uploaded=False, cloud_run_started=False,
        template_hashes={p.name:sha256(p) for p in HERE.glob('*.py')}))
    print('Built separate Submission23 bundle; serving validation and sealing still required.', flush=True)


if __name__ == '__main__':
    main()
