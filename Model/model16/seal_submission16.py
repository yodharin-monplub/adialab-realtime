"""Verify and archive the prepared bundle; no authentication or network calls."""
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "submissions/submission16"


def main():
    sys.path.insert(0, str(BUNDLE))
    from solution16 import validate_resources
    validate_resources(BUNDLE / "resources")
    settings = json.loads((ROOT / "models/model16/cloud_run_settings.json").read_text())
    assert settings["forceFirstTrain"] is False and settings["trainFrequency"] == 0
    assert settings["runtimeDefinitionName"] == "gcp-cpu"
    import main as entry
    assert entry.INFER_PARALLELISM == 12
    assert (BUNDLE / "main.py").read_bytes() == (ROOT / "models/model16/submission_main.py").read_bytes()
    assert (BUNDLE / "solution16.py").read_bytes() == (ROOT / "models/model16/submission_solution.py").read_bytes()
    secret_match = re.search(r"Crunch_Api_key\s*=\s*(\S+)", (ROOT / "AGENTS.md").read_text())
    secret = secret_match.group(1).encode() if secret_match else None
    entries = {}
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(BUNDLE).as_posix()
        if relative == "bundle_manifest.json":
            continue
        assert not path.is_symlink(), f"Unexpected symlink: {relative}"
        assert path.name not in {"AGENTS.md", ".env", "token"}, f"Unexpected file: {relative}"
        data = path.read_bytes()
        assert not secret or secret not in data, f"Credential detected: {relative}"
        entries[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = BUNDLE / "bundle_manifest.json"
    manifest.write_text(json.dumps(dict(submission="submission_16", files=entries), indent=2) + "\n")
    archive = ROOT / "submissions/submission16.zip"
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name in [*entries, "bundle_manifest.json"]:
            output.write(BUNDLE / name, name)
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        for name, spec in entries.items():
            assert hashlib.sha256(output.read(name)).hexdigest() == spec["sha256"]
    receipt = dict(archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   files=len(entries) + 1, credentials_excluded=True,
                   resource_hashes_valid=True, uploaded=False, cloud_run_started=False)
    (ROOT / "models/model16/submission_validation/package.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
