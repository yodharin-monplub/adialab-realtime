"""Build Model26 separately from the sealed cloud-confirmed Model23."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.model26.deployment.check_candidate import check
from models.model26.run_model26 import frozen_study, PARENT
from models.model11.pipeline import sha256, write_json

HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'submissions/submission23'
DEST = ROOT/'submissions/submission26'
OUTPUT = ROOT/'models/model26/deployment_validation'


def main():
    if DEST.exists():
        raise FileExistsError('Refusing to overwrite Submission26')
    evidence = check()
    assert evidence['ready']
    run, protocol, fingerprint = frozen_study()
    assert sha256(SOURCE/'bundle_manifest.json') == '7d84dafdcb3ac317b4f189bd1b62bdd511df9732989fdfbc03f7d3c88be2130a'
    sealed = json.loads((SOURCE/'bundle_manifest.json').read_text())
    for name, record in sealed['files'].items():
        path = SOURCE/name
        assert not path.is_symlink() and path.resolve().is_relative_to(SOURCE.resolve())
        assert sha256(path) == record['sha256']
    manifest = json.loads((SOURCE/'resources/manifest.json').read_text())
    core = next(c for c in manifest['feature_components'] if c['name'] == 'core__feature__xgboost__fisher_256__xgb_depth3')
    parent = json.loads((PARENT/'protocol.json').read_text())
    assert len(core['voters']) == 5
    stage = Path(tempfile.mkdtemp(prefix='submission26-staging-', dir=ROOT/'submissions'))
    for name in sealed['files']:
        if name in ('main.py', 'solution23.py', 'pairing.py', 'readme.txt'):
            continue
        target = stage/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE/name, target)
    for fold, voter in enumerate(core['voters']):
        assert voter['feature_names'] == parent['contract']['names'][fold]
        for key, arm in [('depth25_path', 'aligned_25'), ('depth60_path', 'aligned_60')]:
            name = f'feature_models/{core["name"]}/depth4_{arm}_fold{fold}.ubj'
            shutil.copy2(run/f'{arm}_fold{fold}.ubj', stage/'resources'/name)
            voter[key] = name
            manifest['file_sha256'][name] = sha256(stage/'resources'/name)
    manifest.update(submission='submission_26', backbone_augmentation='depth_mix',
        training_protocol_sha256=fingerprint, study_summary_sha256=sha256(run/'summary.json'),
        pairing_recipe='Nested float32 equal averages: depth3 pair and depth4 pair, then five-fold logit bag',
        validation_note='Exploratory CV0.632406979; cloud confirmation required.')
    write_json(stage/'resources/manifest.json', manifest)
    for source, target in [('entry.py','main.py'), ('solution.py','solution26.py'), ('pairing.py','pairing.py'), ('readme.txt','readme.txt')]:
        shutil.copy2(HERE/source, stage/target)
    stage.rename(DEST)
    OUTPUT.mkdir(exist_ok=True)
    settings = json.loads((ROOT/'models/model23/deployment_validation/cloud_run_settings.json').read_text())
    settings.pop('submissionId', None)
    settings['name'] = 'Submission26 — depth diversity, pretrained inference only'
    assert settings['forceFirstTrain'] is False and settings['trainFrequency'] == 0
    write_json(OUTPUT/'cloud_run_settings.json', settings)
    write_json(OUTPUT/'build.json', dict(evidence=evidence, sealed=False, uploaded=False,
        forceFirstTrain=False, infer_parallelism=12, original_model_resources_unchanged=True))
    print('Built separate Submission26; parity, benchmark and sealing pending.')


if __name__ == '__main__':
    main()
