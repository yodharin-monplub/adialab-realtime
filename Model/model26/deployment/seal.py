"""Seal the validated bundle and estimate cloud cost before uploading."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT/'submissions/submission26'
OUTPUT = ROOT/'models/model26/deployment_validation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(BUNDLE))
    from solution26 import validate_resources
    validate_resources(BUNDLE/'resources')
    parity = json.loads((OUTPUT/'parity.json').read_text())
    assert parity['final_probability_max_error'] <= 1e-7
    assert parity['repeated_inference_bitwise_identical'] and parity['incremental_consumption_passed'] and parity['train_does_not_consume_data']
    smoke = json.loads((OUTPUT/'cloud_smoke.json').read_text())
    assert smoke['cloud_python_compatible'] and smoke['maximum_error'] <= 1e-7
    baseline = json.loads((OUTPUT/'baseline_benchmark_w2_n24.json').read_text())
    benchmark = json.loads((OUTPUT/'benchmark_w2_n24.json').read_text())
    assert [r['id'] for r in baseline['series_results']] == [r['id'] for r in benchmark['series_results']]
    assert baseline['rows'] == benchmark['rows']
    assert baseline['workers'] == benchmark['workers'] == 2
    reference = json.loads((ROOT/'models/model23/deployment_validation/model20_cloud_reference.json').read_text())
    assert reference['success'] and reference['cpu'] == 16384 and reference['memory'] == 65536
    wall_ratio = benchmark['wall_seconds']/baseline['wall_seconds']
    worker_ratio = benchmark['summed_worker_seconds']/baseline['summed_worker_seconds']
    ratio = max(1., wall_ratio, worker_ratio)
    hours = reference['duration']*ratio/3600
    estimate = dict(nominal_cloud_hours=hours, planning_range_hours=[round(hours*.8, 1), round(hours*1.3, 1)],
        matched_local_time_ratio=ratio, measured_wall_ratio=wall_ratio, measured_worker_time_ratio=worker_ratio,
        ratio_method='Conservative maximum of wall-time ratio, summed-worker-time ratio, and 1.0',
        previous_actual_cloud_seconds=reference['duration'],
        peak_measured_worker_rss_mib=benchmark['peak_worker_rss_mib'],
        twelve_worker_rss_sum_gib=12*benchmark['peak_worker_rss_mib']/1024,
        target_cpu=16, target_ram_gib=64, infer_workers=12,
        caveat='Estimate from 24 matched development series with two workers and a previous actual cloud run; excludes queue time, host variance and platform overhead changes. Candidate timing briefly overlapped compatibility/unit checks and may be conservatively biased. Worker RSS extrapolation excludes parent and platform overhead.')
    (OUTPUT/'cloud_estimate.json').write_text(json.dumps(estimate, indent=2)+'\n')
    settings = json.loads((OUTPUT/'cloud_run_settings.json').read_text())
    assert settings['forceFirstTrain'] is False and settings['trainFrequency'] == 0
    import main as entry
    assert entry.INFER_PARALLELISM == 12
    for source, target in (('entry.py','main.py'),('solution.py','solution26.py'),('pairing.py','pairing.py'),('readme.txt','readme.txt')):
        assert (HERE/source).read_bytes() == (BUNDLE/target).read_bytes()
    secret = re.search(r'Crunch_Api_key\s*=\s*(\S+)', (ROOT/'AGENTS.md').read_text()).group(1).encode()
    entries = {}
    for path in sorted(BUNDLE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc' or path.name == 'bundle_manifest.json':
            continue
        name = path.relative_to(BUNDLE).as_posix()
        assert not path.is_symlink() and path.name not in {'AGENTS.md', '.env', 'token'}
        data = path.read_bytes()
        assert secret not in data, 'Credential detected in bundle'
        entries[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (BUNDLE/'bundle_manifest.json').write_text(json.dumps(dict(submission='submission_26', files=entries),indent=2)+'\n')
    from crunch.command.push import list_code_files, list_model_files
    os.chdir(BUNDLE)
    code, models = list(list_code_files(str(BUNDLE),'resources')), list(list_model_files(str(BUNDLE),'resources'))
    actual = {n for _, n in code} | {'resources/'+n for _, n in models}
    assert actual == set(entries) | {'bundle_manifest.json'}
    archive = ROOT/'submissions/submission26.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name in sorted(actual):
            output.write(BUNDLE/name, name)
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        for name, record in entries.items():
            assert hashlib.sha256(output.read(name)).hexdigest() == record['sha256']
    package = dict(archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size,
        sha256=sha(archive), code_files=len(code), model_files=len(models), credentials_excluded=True,
        hashes_verified=True, uploaded=False, cloud_run_started=False)
    (OUTPUT/'package.json').write_text(json.dumps(package, indent=2)+'\n')
    print(json.dumps(dict(package=package, cloud_estimate=estimate), indent=2), flush=True)


if __name__ == '__main__':
    main()
