"""Build an isolated Model73 submission from the sealed Model26 package."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'submissions/submission26'
DEST = ROOT / 'submissions/submission27'
EVIDENCE = ROOT / 'models/model73/deployment_validation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    if DEST.exists():
        raise FileExistsError('Submission27 already exists; refusing to overwrite it')
    base_seal = json.loads((SOURCE / 'bundle_manifest.json').read_text())
    for name, record in base_seal['files'].items():
        path = SOURCE / name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError(f'Unsafe base file: {name}')
        if sha(path) != record['sha256']:
            raise ValueError(f'Changed sealed Model26 file: {name}')
    latest = json.loads((ROOT / 'models/model62/runs/latest.json').read_text())
    run = ROOT / latest['directory']
    if sha(run / 'protocol.json') != latest['protocol_sha256']:
        raise ValueError('Changed frozen Model62 protocol')
    protocol = json.loads((run / 'protocol.json').read_text())
    names = protocol['contract']['names']
    if len(names) != 5 or any(len(item) != 317 for item in names):
        raise ValueError('Changed Model62 feature contract')
    manifest = json.loads((SOURCE / 'resources/manifest.json').read_text())
    if manifest.get('submission') != 'submission_26':
        raise ValueError('Expected sealed Model26 base')
    if any(any(feature not in manifest['feature_names'] for feature in fold[:-61])
           for fold in names):
        raise ValueError('Model62 feature absent from deployed detector union')
    from models.model57.whitening import FEATURE_NAMES
    if any(fold[-61:] != list(FEATURE_NAMES) for fold in names):
        raise ValueError('Changed Model62 whitening features')
    stage = Path(tempfile.mkdtemp(prefix='submission27-staging-', dir=ROOT / 'submissions'))
    for name in base_seal['files']:
        if name in ('main.py', 'readme.txt'):
            continue
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, destination)
    voters = []
    for fold in range(5):
        source = run / f'lightgbm_aligned_40_fold{fold}.txt'
        checkpoint = json.loads((run / f'lightgbm_aligned_40_fold{fold}.json').read_text())
        if checkpoint['artifacts'][source.name] != sha(source):
            raise ValueError(f'Model62 fold {fold} model checkpoint changed')
        relative = f'model73_lightgbm/fold_{fold}.txt'
        destination = stage / 'resources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest['file_sha256'][relative] = sha(destination)
        voters.append(dict(fold=fold, path=relative, names=names[fold]))
    manifest['cloud_candidate'] = 'model73_direct_model62'
    manifest['model73_white_voter'] = dict(
        replaces='core__feature__xgboost__fisher_256__xgb_depth3',
        feature_names=list(FEATURE_NAMES), voters=voters,
        protocol_sha256=latest['protocol_sha256'])
    manifest['validation_note'] = ('Exploratory Model73 direct-voter OOF 0.6354292998; '
                                   'cloud confirmation required; not Model62 full research recipe.')
    write_json(stage / 'resources/manifest.json', manifest)
    for source, target in (('entry.py', 'main.py'), ('solution.py', 'solution73.py'),
                           ('readme.txt', 'readme.txt')):
        shutil.copy2(HERE / source, stage / target)
    shutil.copy2(ROOT / 'models/model57/whitening.py', stage / 'model73_whitening.py')
    stage.rename(DEST)
    EVIDENCE.mkdir(exist_ok=True)
    settings = json.loads((ROOT / 'models/model26/deployment_validation/cloud_run_settings.json').read_text())
    settings.pop('submissionId', None)
    settings['name'] = 'Submission27 — Model73 direct LightGBM voter, pretrained inference'
    if settings['forceFirstTrain'] is not False or settings['trainFrequency'] != 0:
        raise ValueError('Cloud training must be disabled')
    write_json(EVIDENCE / 'cloud_run_settings.json', settings)
    write_json(EVIDENCE / 'build.json', dict(source='sealed_submission26',
        model62_protocol_sha256=latest['protocol_sha256'],
        model73_bundle='submissions/submission27', uploaded=False, cloud_run_started=False))
    print('Built Submission27 from sealed Model26 plus five frozen Model62 LightGBM voters.')


if __name__ == '__main__':
    main()
