"""Build a new bundle from verified Model16 assets; change only XGBoost voters."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model18.run_model18 import read_checkpoint
from models.model11.pipeline import sha256, write_json

HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'submissions/submission16'
DESTINATION = ROOT/'submissions/submission20'


def main():
    if DESTINATION.exists():
        raise FileExistsError(f'Refusing to overwrite {DESTINATION}')
    latest = json.loads((HERE/'runs/latest.json').read_text())
    run = ROOT/latest['directory']
    protocol_sha = sha256(run/'protocol.json')
    assert protocol_sha == latest['protocol_sha256']
    protocol = json.loads((run/'protocol.json').read_text())
    result = json.loads((run/'summary.json').read_text())
    assert result['selected_candidate'] == 'aligned_25'
    assert result['bootstrap_replicates'] == 1000 and result['results']['aligned_25']['passes'] is True
    sealed = json.loads((SOURCE/'bundle_manifest.json').read_text())
    for name, record in sealed['files'].items():
        assert sha256(SOURCE/name) == record['sha256'], name
    for fold in range(5):
        checkpoint = read_checkpoint(run/f'aligned_25_fold{fold}.npz',
            dict(protocol_sha256=protocol_sha, fold=fold, arm='aligned_25'))
        assert checkpoint is not None
    shutil.copytree(SOURCE, DESTINATION, ignore=shutil.ignore_patterns('__pycache__', '*.pyc',
        'bundle_manifest.json', 'main.py', 'solution16.py', 'readme.txt'))
    resources = DESTINATION/'resources'
    manifest = json.loads((resources/'manifest.json').read_text())
    component = next(c for c in manifest['feature_components']
        if c['name'] == 'core__feature__xgboost__fisher_256__xgb_depth3')
    replaced = []
    for fold, voter in enumerate(component['voters']):
        assert voter['feature_names'] == protocol['contract']['names'][fold]
        assert voter['path'].endswith(f'/fold_{fold}.ubj')
        shutil.copy2(run/f'aligned_25_fold{fold}.ubj', resources/voter['path'])
        manifest['file_sha256'][voter['path']] = sha256(resources/voter['path'])
        replaced.append(voter['path'])
    manifest.update(submission='submission_20', backbone_augmentation='aligned_25',
        training_protocol_sha256=protocol_sha, study_summary_sha256=sha256(run/'summary.json'),
        validation_note='Local OOF 0.631575537 is exploratory; it does not validate the five-voter deployment bag or predict the cloud score.')
    write_json(resources/'manifest.json', manifest)
    for source, target in (('submission_main.py', 'main.py'), ('submission_solution.py', 'solution20.py'),
                           ('submission_readme.txt', 'readme.txt')):
        shutil.copy2(HERE/source, DESTINATION/target)
    settings = json.loads((ROOT/'models/model16/cloud_run_settings.json').read_text())
    settings.pop('submissionId', None)
    settings['name'] = 'Submission 20 — aligned-splice XGBoost backbone, pretrained inference only'
    assert settings['forceFirstTrain'] is False and settings['trainFrequency'] == 0
    write_json(HERE/'cloud_run_settings.json', settings)
    output = HERE/'deployment_validation'
    output.mkdir(exist_ok=True)
    write_json(output/'build.json', dict(base_bundle_sha256=sha256(SOURCE/'bundle_manifest.json'),
        replaced_resource_files=replaced, unchanged_resource_files=len(manifest['file_sha256'])-len(replaced),
        model20_run=latest['directory'], no_training_or_validation_data_bundled=True,
        forceFirstTrain=False, infer_parallelism=12, cloud_run_started=False))
    print(f'Prepared Submission20: replaced {len(replaced)} XGBoost voters; other model resources unchanged', flush=True)


if __name__ == '__main__':
    main()
