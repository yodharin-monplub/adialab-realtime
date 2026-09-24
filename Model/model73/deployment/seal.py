"""Seal Model73 only after parity, cloud compatibility and matched timing pass."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / 'submissions/submission27'
OUTPUT = ROOT / 'models/model73/deployment_validation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    if (BUNDLE / 'bundle_manifest.json').exists() or (ROOT / 'submissions/submission27.zip').exists():
        raise FileExistsError('Submission27 already sealed; refusing to overwrite')
    sys.path.insert(0, str(BUNDLE))
    from solution73 import validate_resources
    validate_resources(BUNDLE / 'resources')
    checkpoint = json.loads((OUTPUT / 'checkpoint_parity.json').read_text())
    parity = json.loads((OUTPUT / 'parity.json').read_text())
    smoke = json.loads((OUTPUT / 'cloud_smoke.json').read_text())
    baseline = json.loads((OUTPUT / 'baseline_benchmark_w2_n24.json').read_text())
    candidate = json.loads((OUTPUT / 'benchmark_w2_n24.json').read_text())
    if checkpoint['maximum_error'] > 1e-7 or parity['final_probability_max_error'] > 1e-7:
        raise ValueError('Candidate failed model or inference parity')
    if not (parity['repeated_inference_bitwise_identical'] and
            parity['incremental_consumption_passed'] and parity['train_does_not_consume_data']):
        raise ValueError('Candidate failed streaming or inference-only checks')
    if not smoke['cloud_python_compatible'] or smoke['maximum_error'] > 1e-7:
        raise ValueError('Candidate failed Python 3.12 smoke test')
    if [item['id'] for item in baseline['series_results']] != [
            item['id'] for item in candidate['series_results']]:
        raise ValueError('Benchmark series mismatch')
    if baseline['rows'] != candidate['rows'] or baseline['workers'] != candidate['workers'] != 2:
        raise ValueError('Benchmark row or worker mismatch')
    ratio = max(1.0, candidate['wall_seconds'] / baseline['wall_seconds'],
                candidate['summed_worker_seconds'] / baseline['summed_worker_seconds'])
    previous_cloud_seconds = 20237.0  # Verified Model26 run 114949.
    nominal_hours = previous_cloud_seconds * ratio / 3600
    estimate = dict(nominal_cloud_hours=nominal_hours,
                    planning_range_hours=[round(nominal_hours * .8, 1), round(nominal_hours * 1.3, 1)],
                    matched_local_ratio=ratio,
                    baseline_wall_seconds=baseline['wall_seconds'],
                    candidate_wall_seconds=candidate['wall_seconds'],
                    baseline_worker_seconds=baseline['summed_worker_seconds'],
                    candidate_worker_seconds=candidate['summed_worker_seconds'],
                    peak_worker_rss_mib=candidate['peak_worker_rss_mib'],
                    twelve_worker_rss_sum_gib=12 * candidate['peak_worker_rss_mib'] / 1024,
                    previous_cloud_run_id=114949,
                    previous_cloud_seconds=previous_cloud_seconds,
                    infer_workers=12, target_cpu=16, target_ram_gib=64,
                    caveat='Matched 24-series local benchmark extrapolated from Model26 cloud run; queue, platform, and series-length distribution can change runtime. Twelve-worker RSS excludes parent/platform overhead.')
    write_json(OUTPUT / 'cloud_estimate.json', estimate)
    if nominal_hours > 9.0 or estimate['twelve_worker_rss_sum_gib'] > 48:
        raise ValueError('Model73 estimated cloud runtime or RAM exceeds submission safety gate')
    settings = json.loads((OUTPUT / 'cloud_run_settings.json').read_text())
    if settings['forceFirstTrain'] is not False or settings['trainFrequency'] != 0:
        raise ValueError('Cloud training is not disabled')
    from main import INFER_PARALLELISM
    if INFER_PARALLELISM != 12:
        raise ValueError('Unexpected cloud inference parallelism')
    for source, target in (('entry.py', 'main.py'), ('solution.py', 'solution73.py'),
                           ('readme.txt', 'readme.txt')):
        if (HERE / source).read_bytes() != (BUNDLE / target).read_bytes():
            raise ValueError(f'Changed deployment source: {source}')
    if (ROOT / 'models/model57/whitening.py').read_bytes() != (BUNDLE / 'model73_whitening.py').read_bytes():
        raise ValueError('Changed frozen whitening implementation')
    from dotenv import dotenv_values
    secret = dotenv_values(ROOT / '.env').get('CRUNCH_API_KEY')
    if not secret:
        raise ValueError('Missing cloud credential')
    entries = {}
    for path in sorted(BUNDLE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc' or path.name == 'bundle_manifest.json':
            continue
        if path.is_symlink() or path.name in {'AGENTS.md', '.env', 'token'}:
            raise ValueError(f'Unsafe submission file: {path}')
        name = path.relative_to(BUNDLE).as_posix()
        data = path.read_bytes()
        if secret.encode() in data:
            raise ValueError('Credential detected in submission bundle')
        entries[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    write_json(BUNDLE / 'bundle_manifest.json', dict(submission='submission_27', files=entries))
    from crunch.command.push import list_code_files, list_model_files
    os.chdir(BUNDLE)
    code = list(list_code_files(str(BUNDLE), 'resources'))
    models = list(list_model_files(str(BUNDLE), 'resources'))
    actual = {name for _, name in code} | {'resources/' + name for _, name in models}
    if actual != set(entries) | {'bundle_manifest.json'}:
        raise ValueError('Cloud upload inventory differs from sealed bundle')
    archive = ROOT / 'submissions/submission27.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted(actual):
            zipped.write(BUNDLE / name, name)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Corrupt submission archive')
        for name, record in entries.items():
            if hashlib.sha256(zipped.read(name)).hexdigest() != record['sha256']:
                raise ValueError(f'Archive mismatch: {name}')
    package = dict(archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size,
                   sha256=sha(archive), code_files=len(code), model_files=len(models),
                   credentials_excluded=True, hashes_verified=True,
                   uploaded=False, cloud_run_started=False)
    write_json(OUTPUT / 'package.json', package)
    print(json.dumps(dict(package=package, cloud_estimate=estimate), indent=2), flush=True)


if __name__ == '__main__':
    main()
