"""Build and inspect Submission29 locally. Never submits or starts cloud jobs."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = ROOT / "submissions/submission29"


def main():
    BUNDLE.mkdir(exist_ok=True)
    reference = BUNDLE / "resources/X_test.reduced.parquet"
    provenance = json.loads((HERE / "reference_source.json").read_text())
    if hashlib.sha256(reference.read_bytes()).hexdigest() != provenance["sha256"]:
        raise ValueError("Reference data changed")
    for name in ("main.py", "overlap_probe.py"):
        shutil.copyfile(HERE / "deployment" / name, BUNDLE / name)
    shutil.copyfile(HERE / "readme.txt", BUNDLE / "readme.txt")
    shutil.copyfile(HERE / "reference_source.json", BUNDLE / "resources/reference_source.json")
    (BUNDLE / "requirements.txt").write_text("numpy>=2,<3\npandas>=2,<3\npyarrow>=18\n")
    (BUNDLE / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    entries = {}
    for name in ("main.py", "overlap_probe.py", "readme.txt", "requirements.txt", ".gitignore",
                 "resources/X_test.reduced.parquet", "resources/reference_source.json"):
        data = (BUNDLE / name).read_bytes()
        entries[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (BUNDLE / "bundle_manifest.json").write_text(json.dumps({"submission": "submission29",
        "model": "model108", "diagnostic_only": True, "files": entries}, indent=2) + "\n")
    # These SDK helpers enumerate files only; they make no upload/API calls.
    from crunch.command.push import list_code_files, list_model_files
    code = list(list_code_files(str(BUNDLE), "resources"))
    models = list(list_model_files(str(BUNDLE), "resources"))
    actual = {name for _, name in code} | {"resources/" + name for _, name in models}
    if actual != set(entries) | {"bundle_manifest.json"}:
        raise ValueError(f"Unexpected upload inventory: {actual.symmetric_difference(set(entries) | {'bundle_manifest.json'})}")
    archive = ROOT / "submissions/submission29.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(actual):
            zipped.write(BUNDLE / name, name)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name, record in entries.items():
            assert hashlib.sha256(zipped.read(name)).hexdigest() == record["sha256"]
    package = {"archive": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "uploaded": False,
        "cloud_run_started": False, "code_files": len(code), "model_files": len(models)}
    (HERE / "package.json").write_text(json.dumps(package, indent=2) + "\n")
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
