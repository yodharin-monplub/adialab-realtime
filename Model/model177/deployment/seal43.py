"""Seal submissions/submission43 (Submission42 + Model177 lattice gate): manifest + zip + credential scan.
Never uploads.  Refuses unless Model/model177/validation_100.json exists (validate177.py passed: every
non-lattice local series bit-identical to Submission42, both local lattice series fire at tau) and the
bundle's untouched files are byte-identical to Submission42's.
Usage: python Model/model177/deployment/seal43.py
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
NAME = 'submission43'
BUNDLE = ROOT / 'submissions' / NAME
PARENT = ROOT / 'submissions' / 'submission42'
MODEL = ROOT / 'Model/model177'
OUTPUT = MODEL / 'deployment_validation'
ARCHIVE = ROOT / 'submissions' / f'{NAME}.zip'


def main():
    if ARCHIVE.exists():
        raise FileExistsError(f'{ARCHIVE.name} already exists; refusing to overwrite')
    validation = json.loads((MODEL / 'validation_100.json').read_text())
    if len(validation['ids']) != 100 or sorted(validation['lattice']) != [10044, 10079]:
        raise ValueError('validation_100.json is not the full local-test validation')
    for sid, rec in validation['series'].items():
        if rec['first_1.0'] != rec['tau_pos']:
            raise ValueError(f'lattice series {sid} does not fire at tau in the validation')
    if validation['tsauc43'] <= validation['tsauc42']:
        raise ValueError('Submission43 does not beat Submission42 on the local test set')
    # the gate must be the validated source; everything else must be Submission42's bytes
    for name, source in (('lattice_gate.py', MODEL / 'lattice_gate.py'),):
        if (BUNDLE / name).read_bytes() != source.read_bytes():
            raise ValueError(f'Packaged {name} differs from Model/model177/{name}')
    changed = {'main.py', 'readme.txt', 'lattice_gate.py', 'bundle_manifest.json'}
    for path in sorted(BUNDLE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        rel = path.relative_to(BUNDLE).as_posix()
        if rel in changed:
            continue
        twin = PARENT / rel
        if not twin.is_file() or twin.read_bytes() != path.read_bytes():
            raise ValueError(f'{rel} differs from Submission42')
    sys.path.insert(0, str(BUNDLE))
    from solution147 import validate_resources
    validate_resources(BUNDLE / 'resources')
    secrets = [line.split('=', 1)[1].strip().strip('"').strip("'")
               for line in (ROOT / 'Other/.env').read_text().splitlines() if '=' in line and line.split('=', 1)[1].strip()]
    email = ROOT / 'Other/email.txt'
    if email.exists():
        secrets += [line.strip() for line in email.read_text().splitlines() if len(line.strip()) > 8]
    entries = {}
    for path in sorted(BUNDLE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc' or path.name == 'bundle_manifest.json':
            continue
        if path.is_symlink() or path.name in {'AGENTS.md', '.env', 'token', 'email.txt'}:
            raise ValueError(f'Unsafe submission file: {path}')
        data = path.read_bytes()
        if any(s and len(s) > 8 and s.encode() in data for s in secrets):
            raise ValueError(f'Credential detected in {path}')
        entries[path.relative_to(BUNDLE).as_posix()] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (BUNDLE / 'bundle_manifest.json').write_text(json.dumps({'submission': NAME, 'files': entries}, indent=2) + '\n')
    names = sorted(entries) + ['bundle_manifest.json']
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in names:
            zipped.write(BUNDLE / name, name)
    with zipfile.ZipFile(ARCHIVE) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Corrupt archive')
    OUTPUT.mkdir(exist_ok=True)
    package = {'archive': f'submissions/{ARCHIVE.name}', 'bytes': ARCHIVE.stat().st_size,
               'sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), 'files': len(names),
               'credentials_excluded': True, 'uploaded': False, 'cloud_run_started': False,
               'validation': {k: validation[k] for k in ('tsauc42', 'tsauc43', 'lattice')},
               'estimated_cloud_hours': 5.0}
    (OUTPUT / f'package_{NAME}.json').write_text(json.dumps(package, indent=2) + '\n')
    print(json.dumps({k: package[k] for k in ('archive', 'bytes', 'files', 'uploaded')}, indent=2))


if __name__ == '__main__':
    main()
