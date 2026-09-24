"""Seal Submission33 after validate33.py passed: file manifest + zip.  Never uploads.

Refuses unless models/model147/deployment_validation/validation33.json shows predictions equal to
the holdout run (float32 rounding), a bit-identical replay and an estimated run time with margin.
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / "submissions/submission33"
OUTPUT = ROOT / "models/model147/deployment_validation"
ARCHIVE = ROOT / "submissions/submission33.zip"


def main():
    if (BUNDLE / "bundle_manifest.json").exists() and json.loads((BUNDLE / "bundle_manifest.json").read_text()).get("submission") == "submission_33":
        raise FileExistsError("Submission33 is already sealed")
    if ARCHIVE.exists():
        raise FileExistsError("submission33.zip already exists; refusing to overwrite")
    v = json.loads((OUTPUT / "validation33.json").read_text())
    if not (v["max_abs_diff_vs_holdout_run"] < 1e-6 and v["deterministic_replay_max_abs_diff"] <= 1e-8
            and v["estimated_cloud_hours"] * 1.3 < 9.0):
        raise ValueError("validation33.json does not meet the sealing conditions")
    sys.path.insert(0, str(BUNDLE))
    from solution147 import validate_resources
    validate_resources(BUNDLE / "resources")
    for name in ("solution147.py", "split_tests_numpy.py", "main.py"):
        if (HERE / name).read_bytes() != (BUNDLE / name).read_bytes():
            raise ValueError(f"Packaged {name} differs from the validated source")
    secrets = [line.split("=", 1)[1].strip().strip('"').strip("'") for line in (ROOT / ".env").read_text().splitlines()
               if "=" in line and line.split("=", 1)[1].strip()]
    entries = {}
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == "bundle_manifest.json":
            continue
        if path.is_symlink() or path.name in {"AGENTS.md", ".env", "token", "email.txt"}:
            raise ValueError(f"Unsafe submission file: {path}")
        data = path.read_bytes()
        if any(s and len(s) > 8 and s.encode() in data for s in secrets):
            raise ValueError(f"Credential detected in {path}")
        entries[path.relative_to(BUNDLE).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (BUNDLE / "bundle_manifest.json").write_text(json.dumps({"submission": "submission_33", "files": entries}, indent=2) + "\n")
    names = sorted(entries) + ["bundle_manifest.json"]
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in names:
            zipped.write(BUNDLE / name, name)
    with zipfile.ZipFile(ARCHIVE) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Corrupt archive")
    package = {"archive": "submissions/submission33.zip", "bytes": ARCHIVE.stat().st_size,
               "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), "files": len(names),
               "credentials_excluded": True, "uploaded": False, "cloud_run_started": False, "validation": v}
    (OUTPUT / "package33.json").write_text(json.dumps(package, indent=2) + "\n")
    print(json.dumps({k: package[k] for k in ("archive", "bytes", "files", "uploaded")}, indent=2))


if __name__ == "__main__":
    main()
