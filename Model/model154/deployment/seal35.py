"""Seal a validated Model154 bundle: file manifest + zip, credential scan.  Never uploads.
Refuses unless deployment_validation/validation_{BUNDLE}.json shows offline parity (< 1e-6), a bit-identical
replay and a run time well inside the weekly limit.
Usage: python models/model154/deployment/seal35.py [BUNDLE]      (default submission35)
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SERVE = ROOT / "models/model147/deployment"
NAME = sys.argv[1] if len(sys.argv) > 1 else "submission35"
BUNDLE = ROOT / "submissions" / NAME
OUTPUT = ROOT / ("models/model169/deployment_validation" if NAME in ("submission41", "submission42") else "models/model154/deployment_validation")
ARCHIVE = ROOT / "submissions" / f"{NAME}.zip"


def main():
    if ARCHIVE.exists():
        raise FileExistsError(f"{ARCHIVE.name} already exists; refusing to overwrite")
    v = json.loads((OUTPUT / f"validation_{NAME}.json").read_text())
    if not (v["max_abs_diff_vs_offline"] < 2e-6 and v["deterministic_replay_max_abs_diff"] <= 1e-8
            and v["estimated_cloud_hours"] * 1.3 < 12.0):
        raise ValueError("validation file does not meet the sealing conditions")
    sys.path.insert(0, str(BUNDLE))
    from solution147 import validate_resources
    validate_resources(BUNDLE / "resources")
    for name in ("solution147.py", "split_tests_numpy.py", "main.py"):
        if (SERVE / name).read_bytes() != (BUNDLE / name).read_bytes():
            raise ValueError(f"Packaged {name} differs from the validated source")
    secrets = [line.split("=", 1)[1].strip().strip('"').strip("'") for line in (ROOT / ".env").read_text().splitlines()
               if "=" in line and line.split("=", 1)[1].strip()]
    secrets += [line.strip() for line in (ROOT / "email.txt").read_text().splitlines() if len(line.strip()) > 8]
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
    (BUNDLE / "bundle_manifest.json").write_text(json.dumps({"submission": NAME, "files": entries}, indent=2) + "\n")
    names = sorted(entries) + ["bundle_manifest.json"]
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in names:
            zipped.write(BUNDLE / name, name)
    with zipfile.ZipFile(ARCHIVE) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Corrupt archive")
    package = {"archive": f"submissions/{ARCHIVE.name}", "bytes": ARCHIVE.stat().st_size,
               "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), "files": len(names),
               "credentials_excluded": True, "uploaded": False, "cloud_run_started": False, "validation": v}
    (OUTPUT / f"package_{NAME}.json").write_text(json.dumps(package, indent=2) + "\n")
    print(json.dumps({k: package[k] for k in ("archive", "bytes", "files", "uploaded")}, indent=2))


if __name__ == "__main__":
    main()

