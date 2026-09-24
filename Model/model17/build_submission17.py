"""Bundle saved models only; no fitting, label access, upload, or cloud run."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "submissions/submission16"
OLD = ROOT / "models/model16/submission_validation/baseline_resources"
DESTINATION = ROOT / "submissions/submission17"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite {DESTINATION}")
    study_dir = ROOT / json.loads((ROOT / "models/model17/runs/latest.json").read_text())["directory"]
    recipe = json.loads((study_dir / "frozen_candidate.json").read_text())
    assert recipe["weights"] == [.85, 0., .15]
    sealed = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, spec in sealed["files"].items():
        assert sha256(SOURCE / name) == spec["sha256"], name
    old = json.loads((OLD / "manifest.json").read_text())
    for name, expected in old["file_sha256"].items():
        assert sha256(OLD / name) == expected, name
    shutil.copytree(SOURCE, DESTINATION, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "bundle_manifest.json", "readme.txt", "main.py"))
    resources = DESTINATION / "resources"
    shared = 0
    for name, expected in old["file_sha256"].items():
        path = resources / name
        if path.exists():
            assert sha256(path) == expected, name
            shared += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OLD / name, path)
    # Retain runtime contracts, not historical holdout score metadata.
    kept = ("union_banks", "detector_families", "detector_ensemble", "feature_components", "meta_models",
            "components", "global_recipe", "global_transform", "file_sha256", "deployment_profile", "excluded_timeout_component")
    ensemble = dict(submission="submission_17", weights=[.85, .15],
                    model16_manifest_sha256=sha256(resources / "manifest.json"),
                    frozen_candidate_sha256=sha256(study_dir / "frozen_candidate.json"),
                    model6={k: old[k] for k in kept}, shared_resource_files=shared,
                    validation="Exploratory OOF screen, not independent validation of deployed five-voter bags.")
    (resources / "ensemble17.json").write_text(json.dumps(ensemble, indent=2) + "\n")
    for source, target in (("submission_main.py", "main.py"), ("submission_solution.py", "solution17.py"), ("submission_readme.txt", "readme.txt")):
        shutil.copy2(ROOT / "models/model17" / source, DESTINATION / target)
    settings = json.loads((ROOT / "models/model16/cloud_run_settings.json").read_text())
    settings.pop("submissionId", None)
    settings["name"] = "Submission 17 — pretrained 85/15 Model16/6"
    (ROOT / "models/model17/cloud_run_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Prepared {DESTINATION}: {sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file())/1024**2:.2f} MiB; {shared} shared resource files", flush=True)


if __name__ == "__main__":
    main()
