"""Seal Submission28 after exact parity and matched runtime checks pass."""
from pathlib import Path
import hashlib
import json
import os
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / "submissions/submission28"
SOURCE = ROOT / "submissions/submission27"
OUTPUT = ROOT / "models/model87/deployment_validation"
ARCHIVE = ROOT / "submissions/submission28.zip"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    if (BUNDLE / "bundle_manifest.json").exists() or ARCHIVE.exists():
        raise FileExistsError("Submission28 is already sealed; refusing to overwrite")

    # Refuse to package a repair made from a changed copy of the failed bundle.
    source_seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in source_seal["files"].items():
        path = SOURCE / name
        if path.is_symlink() or sha(path) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission27 source: {name}")

    sys.path.insert(0, str(BUNDLE))
    from solution73 import validate_resources
    validate_resources(BUNDLE / "resources")

    parity = json.loads((OUTPUT / "parity.json").read_text())
    candidate = json.loads((OUTPUT / "benchmark_w2_n24.json").read_text())
    baseline = json.loads(
        (ROOT / "models/model73/deployment_validation/baseline_benchmark_w2_n24.json").read_text()
    )
    if parity["whitening_max_error"] != 0 or parity["final_max_error"] != 0:
        raise ValueError("Incremental repair is not exactly prediction-equivalent")
    if not (parity["bitwise_identical"] and parity["incremental_consumption"] and
            parity["train_does_not_consume_data"]):
        raise ValueError("Streaming or inference-only validation failed")
    if not parity["python"].startswith("3.12."):
        raise ValueError("Parity was not checked with the cloud Python version")
    if [x["id"] for x in baseline["series_results"]] != [
            x["id"] for x in candidate["series_results"]]:
        raise ValueError("Candidate and baseline benchmark series differ")
    if baseline["rows"] != candidate["rows"] or baseline["workers"] != 2 or candidate["workers"] != 2:
        raise ValueError("Candidate and baseline benchmark workloads differ")

    baseline_worker = baseline["summed_worker_seconds"]
    candidate_worker = candidate["worker_seconds"]
    ratio = max(1.0, candidate["wall_seconds"] / baseline["wall_seconds"],
                candidate_worker / baseline_worker)
    previous_cloud_seconds = 20237.0  # Successful Submission26 cloud run 114949.
    nominal_hours = previous_cloud_seconds * ratio / 3600
    upper_hours = nominal_hours * 1.3
    twelve_worker_rss_gib = 12 * candidate["peak_worker_rss_mib"] / 1024
    estimate = {
        "nominal_cloud_hours": nominal_hours,
        "planning_range_hours": [round(nominal_hours * .8, 1), round(upper_hours, 1)],
        "matched_local_ratio": ratio,
        "baseline_wall_seconds": baseline["wall_seconds"],
        "candidate_wall_seconds": candidate["wall_seconds"],
        "baseline_worker_seconds": baseline_worker,
        "candidate_worker_seconds": candidate_worker,
        "peak_worker_rss_mib": candidate["peak_worker_rss_mib"],
        "twelve_worker_rss_sum_gib": twelve_worker_rss_gib,
        "previous_cloud_run_id": 114949,
        "previous_cloud_seconds": previous_cloud_seconds,
        "failed_cloud_run_id": 116613,
        "failed_cloud_seconds": 35572,
        "infer_workers": 12,
        "target_cpu": 16,
        "target_ram_gib": 64,
        "caveat": "Estimate is anchored to the successful Model26 cloud run and a matched 24-series local benchmark; platform load and hidden series lengths can change runtime."
    }
    write_json(OUTPUT / "cloud_estimate.json", estimate)
    if upper_hours >= 9.0 or twelve_worker_rss_gib >= 48:
        raise ValueError("Estimated runtime or RAM does not have enough platform margin")

    from main import INFER_PARALLELISM
    if INFER_PARALLELISM != 12:
        raise ValueError("Unexpected inference parallelism")
    if (HERE / "solution.py").read_bytes() != (BUNDLE / "solution73.py").read_bytes():
        raise ValueError("Packaged solution differs from the validated source")
    if (HERE / "incremental_whitening.py").read_bytes() != (BUNDLE / "incremental_whitening.py").read_bytes():
        raise ValueError("Packaged incremental whitening differs from the validated source")

    secret = None
    try:
        from dotenv import dotenv_values
        secret = dotenv_values(ROOT / ".env").get("CRUNCH_API_KEY")
    except Exception:
        pass
    entries = {}
    for path in sorted(BUNDLE.rglob("*")):
        if (not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc" or
                path.name == "bundle_manifest.json"):
            continue
        if path.is_symlink() or path.name in {"AGENTS.md", ".env", "token"}:
            raise ValueError(f"Unsafe submission file: {path}")
        name = path.relative_to(BUNDLE).as_posix()
        data = path.read_bytes()
        if secret and secret.encode() in data:
            raise ValueError("Credential detected in submission bundle")
        entries[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    write_json(BUNDLE / "bundle_manifest.json", {"submission": "submission_28", "files": entries})

    from crunch.command.push import list_code_files, list_model_files
    os.chdir(BUNDLE)
    code = list(list_code_files(str(BUNDLE), "resources"))
    models = list(list_model_files(str(BUNDLE), "resources"))
    actual = {name for _, name in code} | {"resources/" + name for _, name in models}
    if actual != set(entries) | {"bundle_manifest.json"}:
        raise ValueError("Cloud upload inventory differs from the sealed bundle")
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted(actual):
            zipped.write(BUNDLE / name, name)
    with zipfile.ZipFile(ARCHIVE) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Corrupt submission archive")
        for name, record in entries.items():
            if hashlib.sha256(zipped.read(name)).hexdigest() != record["sha256"]:
                raise ValueError(f"Archive mismatch: {name}")

    package = {
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "bytes": ARCHIVE.stat().st_size,
        "sha256": sha(ARCHIVE),
        "code_files": len(code),
        "model_files": len(models),
        "credentials_excluded": True,
        "hashes_verified": True,
        "uploaded": False,
        "cloud_run_started": False,
    }
    write_json(OUTPUT / "package.json", package)
    print(json.dumps({"package": package, "cloud_estimate": estimate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
