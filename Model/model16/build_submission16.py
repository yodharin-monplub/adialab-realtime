"""Prepare portable pretrained Submission 16, without uploading or starting a run."""
from pathlib import Path
import hashlib
import importlib
import json
import shutil
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/"models/model16/upstream_cloud_submission9"
DESTINATION = ROOT/"submissions/submission16"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite {DESTINATION}")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(UPSTREAM))
    # The old joblibs wrap backend models in this simple, data-only container.
    # Avoid importing historical training pipelines or requiring missing caches.
    shim = types.ModuleType("models.model6.model6_pipeline")
    shim.FittedModel = type("FittedModel", (), {})
    sys.modules[shim.__name__] = shim
    builder = importlib.import_module("build_resources")
    builder.PROJECT_ROOT = ROOT
    builder.RUN_ROOT = ROOT/"models/model6/runs/tsroc0p64_909b8f5c6c"
    builder.SCREEN_ROOT = ROOT/"models/model5/artifacts/screen"
    from feature_pipeline import feature_names
    stack_path = ROOT/"models/model9/runs/fulldata_819a001db6/pairwise_stack/deployment/coefficients.npz"
    with np.load(stack_path, allow_pickle=False) as payload:
        stack_names = payload["feature_names"].tolist()
    banks = {family:[name.split(":")[-1] for name in stack_names if name.startswith(family+":bank:")]
             for family in ("core", "candidate")}
    union = banks["core"]+banks["candidate"]
    names = feature_names(union)
    resources = DESTINATION/"resources"
    resources.mkdir(parents=True)
    for name in ("feature_pipeline.py", "competition_data.py", "compact_runtime.py"):
        shutil.copy2(UPSTREAM/name, DESTINATION/name)
    shutil.copytree(UPSTREAM/"detector", DESTINATION/"detector", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(UPSTREAM/"solution.py", DESTINATION/"primitive_runtime.py")
    for source, target in (("submission_main.py","main.py"), ("submission_solution.py","solution16.py")):
        shutil.copy2(ROOT/"models/model16"/source, DESTINATION/target)
    shutil.copy2(ROOT/"models/model16/submission_readme.txt", DESTINATION/"readme.txt")
    shutil.copy2(ROOT/"models/model11/features.py", DESTINATION/"compact_features.py")
    frozen = json.loads((builder.RUN_ROOT/"frozen_contract.json").read_text())
    print("Exporting 155 detector voters into native traversal", flush=True)
    _, detector_spec = builder._export_detector_ensemble(resources, frozen, names, banks)
    print("Exporting the three five-voter feature models", flush=True)
    components = builder._export_feature_components(resources, frozen, names)
    shutil.copy2(stack_path, resources/"stack_coefficients.npz")
    latest = json.loads((ROOT/"models/model11/runs/latest.json").read_text())
    compact = []
    (resources/"compact").mkdir()
    for fold in range(5):
        name = f"compact/fold{fold}.txt"
        shutil.copy2(ROOT/latest["directory"]/f"fold{fold}.txt", resources/name)
        compact.append(name)
    component_names = [name.replace(":feature:","__feature__") for name in stack_names if ":feature:" in name]
    # Upstream names already use family__feature__architecture.
    if set(component_names) != {c["name"] for c in components}:
        raise ValueError("Exported feature models differ from the Model 9 stack")
    hashes = {str(p.relative_to(resources)):digest(p) for p in resources.rglob("*") if p.is_file()}
    manifest = dict(submission="submission_16", pretrained_models_bundled=True, union_banks=union,
                    feature_names=names, detector_ensemble=detector_spec, feature_components=components,
                    stack_feature_names=stack_names, stack_feature_components=component_names,
                    compact_models=compact, file_sha256=hashes,
                    recipe="0.81*m9 + 0.09*rolling_max128(m9) + 0.10*m11",
                    deployment_bagging="All five saved development-fold voters; Model 9 all-development stack coefficients.",
                    validation_note="Research OOF TSAUC 0.629300 is exploratory and does not score this five-voter deployment bag.")
    (resources/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (DESTINATION/"requirements.txt").write_text("numpy>=2,<3\nscipy==1.18.0\npandas>=2,<3\npyarrow>=18\nlightgbm==4.6.0\nxgboost==3.4.1\n")
    (DESTINATION/".gitignore").write_text("__pycache__/\n*.pyc\ndata/\nprediction/\ntrain/\n")
    launch = dict(runtimeDefinitionName="gcp-cpu", forceFirstTrain=False, trainFrequency=0,
                  determinismCheck=True, phaseType="SUBMISSION", name="Submission 16 — pretrained inference only")
    (ROOT/"models/model16/cloud_run_settings.json").write_text(json.dumps(launch,indent=2)+"\n")
    print(f"Prepared {DESTINATION}: {sum(p.stat().st_size for p in DESTINATION.rglob('*') if p.is_file())/1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
