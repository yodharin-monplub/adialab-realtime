"""Seal Submission31 after parity and runtime checks pass (never uploads)."""
from pathlib import Path
import hashlib
import json
import os
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SUB = os.environ.get("SUBMISSION_NAME", "submission31")
BUNDLE = ROOT / "submissions" / SUB
OUTPUT = ROOT / "models/model124/deployment_validation" / ("" if SUB == "submission31" else SUB)
ARCHIVE = ROOT / "submissions" / f"{SUB}.zip"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    if (BUNDLE / "bundle_manifest.json").exists() or ARCHIVE.exists():
        raise FileExistsError("Submission31 is already sealed; refusing to overwrite")
    sys.path.insert(0, str(BUNDLE))
    from solution124 import validate_resources
    validate_resources(BUNDLE / "resources")
    parity = json.loads((OUTPUT / "parity.json").read_text())
    bench = json.loads((OUTPUT / "benchmark_w2_n24.json").read_text())
    if not (parity["gru_streaming_vs_offline_max_abs_logit_diff"] < 1e-4 and parity["deterministic_replay"]
            and parity["incremental_consumption"] and parity["train_does_not_consume_data"]
            and parity["python"].startswith("3.12.")):
        raise ValueError("Parity validation incomplete")
    nominal = bench["estimated_cloud_hours"]
    estimate = {"nominal_cloud_hours": nominal, "planning_range_hours": [round(nominal * .8, 1), round(nominal * 1.3, 1)],
                "ratio_vs_submission30_worker_seconds": bench["ratio_worker_seconds"],
                "submission30_actual_hours": 3.93, "infer_workers": 12, "target_cpu": 16, "target_ram_gib": 64}
    write_json(OUTPUT / "cloud_estimate.json", estimate)
    if nominal * 1.3 >= 9.0:
        raise ValueError("Estimated runtime lacks platform margin")
    from main import INFER_PARALLELISM
    if INFER_PARALLELISM != 12:
        raise ValueError("Unexpected inference parallelism")
    for name, source in (("solution124.py", HERE / "solution124.py"), ("main.py", HERE / "main.py"),
                         ("gru_numpy.py", ROOT / "models/model124/gru_numpy.py")):
        if source.read_bytes() != (BUNDLE / name).read_bytes():
            raise ValueError(f"Packaged {name} differs from validated source")
    secret = None
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("CRUNCH_API_KEY="):
            secret = line.split("=", 1)[1].strip().strip('"').strip("'")
    entries = {}
    for path in sorted(BUNDLE.rglob("*")):
        if (not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc"
                or path.name == "bundle_manifest.json"):
            continue
        if path.is_symlink() or path.name in {"AGENTS.md", ".env", "token", "email.txt"}:
            raise ValueError(f"Unsafe submission file: {path}")
        name = path.relative_to(BUNDLE).as_posix()
        data = path.read_bytes()
        if secret and secret.encode() in data:
            raise ValueError("Credential detected in submission bundle")
        entries[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    write_json(BUNDLE / "bundle_manifest.json", {"submission": SUB.replace("submission", "submission_"), "files": entries})
    from crunch.command.push import list_code_files, list_model_files
    os.chdir(BUNDLE)
    code = list(list_code_files(str(BUNDLE), "resources"))
    models = list(list_model_files(str(BUNDLE), "resources"))
    actual = {name.replace("\\", "/") for _, name in code} | {"resources/" + name.replace("\\", "/") for _, name in models}
    if actual != set(entries) | {"bundle_manifest.json"}:
        raise ValueError(f"Cloud upload inventory differs: {sorted(actual ^ (set(entries) | {'bundle_manifest.json'}))[:10]}")
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted(actual):
            zipped.write(BUNDLE / name, name)
    with zipfile.ZipFile(ARCHIVE) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Corrupt submission archive")
        for name, record in entries.items():
            if hashlib.sha256(zipped.read(name)).hexdigest() != record["sha256"]:
                raise ValueError(f"Archive mismatch: {name}")
    package = {"archive": str(ARCHIVE.relative_to(ROOT)), "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE),
               "code_files": len(code), "model_files": len(models), "credentials_excluded": True,
               "hashes_verified": True, "uploaded": False, "cloud_run_started": False}
    write_json(OUTPUT / "package.json", package)
    print(json.dumps({"package": package, "cloud_estimate": estimate}, indent=2))


if __name__ == "__main__":
    main()
