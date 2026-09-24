"""Validate and seal the prepared runtime; do not upload or start a cloud run."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "submissions/submission17"
OUTPUT = ROOT / "models/model17/deployment_validation"


def main():
    sys.path.insert(0, str(BUNDLE))
    from solution17 import validate_resources
    validate_resources(BUNDLE / "resources")
    parity = json.loads((OUTPUT / "parity.json").read_text())
    assert parity["final_max_error"] == 0 and parity["deterministic"] and parity["no_training_data_consumed"]
    json.loads((OUTPUT / "audit.json").read_text())
    for source, target in (("submission_main.py", "main.py"), ("submission_solution.py", "solution17.py"), ("submission_readme.txt", "readme.txt")):
        assert (ROOT / "models/model17" / source).read_bytes() == (BUNDLE / target).read_bytes()
    settings = json.loads((ROOT / "models/model17/cloud_run_settings.json").read_text())
    assert settings["forceFirstTrain"] is False and settings["trainFrequency"] == 0 and "submissionId" not in settings
    import main as entry
    assert entry.INFER_PARALLELISM == 12
    secret = re.search(r"Crunch_Api_key\s*=\s*(\S+)", (ROOT / "AGENTS.md").read_text()).group(1).encode()
    entries = {}
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == "bundle_manifest.json":
            continue
        name = path.relative_to(BUNDLE).as_posix()
        assert not path.is_symlink() and path.name not in {"AGENTS.md", ".env", "token"}
        data = path.read_bytes()
        assert secret not in data, f"Credential found in {name}"
        entries[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (BUNDLE / "bundle_manifest.json").write_text(json.dumps(dict(submission="submission_17", files=entries), indent=2) + "\n")
    # Resolve the actual CLI upload inventory from the correct working directory.
    from crunch.command.push import list_code_files, list_model_files
    os.chdir(BUNDLE)
    code, models = list(list_code_files(str(BUNDLE), "resources")), list(list_model_files(str(BUNDLE), "resources"))
    actual = {name for _, name in code} | {"resources/" + name for _, name in models}
    assert actual == set(entries) | {"bundle_manifest.json"}
    assert not any(name.startswith("resources/") for _, name in code)
    archive = ROOT / "submissions/submission17.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name in actual:
            output.write(BUNDLE / name, name)
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        for name, spec in entries.items():
            assert hashlib.sha256(output.read(name)).hexdigest() == spec["sha256"]
    receipt = dict(archive=str(archive.relative_to(ROOT)), bytes=archive.stat().st_size,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), code_files=len(code), model_files=len(models),
                   credentials_excluded=True, hashes_verified=True, uploaded=False, cloud_run_started=False)
    (OUTPUT / "package.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
