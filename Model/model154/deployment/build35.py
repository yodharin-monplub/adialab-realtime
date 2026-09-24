"""Build Submission35 = sealed Submission31 (chain + GRU ensemble) + the Model154 ALL-data student voters.

Identical serving code to Submission33 (models/model147/deployment: solution147.py, split_tests_numpy.py,
main.py); only the 15 student voter files change (trained on all 10,000 labelled series instead of 3/5 of
the dev series).  Blend constants = BAG3's (models/model112/constants_m147_bag3.json), weight 0.5, as in
the Model152 holdout check that chose this recipe.
Env: ARMS (default ALL,ALLb,ALLc), RUNS (default models/model154/runs), DEST (default submission35).
Never uploads or submits.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SERVE = ROOT / "models/model147/deployment"
ARMS = os.environ.get("ARMS", "ALL,ALLb,ALLc").split(",")
RUNS = ROOT / os.environ.get("RUNS", "models/model154/runs")
DEST = ROOT / "submissions" / os.environ.get("DEST", "submission35")
SOURCE = ROOT / "submissions/submission31"
CONSTANTS = ROOT / "models/model112/constants_m147_bag3.json"
PROTOCOL = ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json"
TRAINING_CODE = [
    "tsauc.py", "models/PIPELINE_SUBMISSION31.txt", "models/SESSION_RESULTS_2026-09-17.md",
    "models/model5/model5_pipeline.py", "models/model10/sources.py", "models/model18/augmentation.py",
    "models/model20/features.py", "models/model20/run_model20.py", "models/model21/run_model21.py",
    "models/model26/run_model26.py", "models/model57/whitening.py", "models/model57/run_model57.py",
    "models/model62/run_model62.py", "models/model84/run_model84.py",
    "models/model109/teacher_features.py", "models/model109/run_model109.py", "models/model111/run_model111.py",
    "models/model112/select_blend.py", "models/model124/export_data.py", "models/model124/train_gru.py",
    "models/model124/select_gru.py", "models/model137/run_model137.py", "models/model140/run_model140.py",
    "models/model144/split_tests.py", "models/model144/run_model144.py",
    "models/model147/build_recuts.py", "models/model147/run_model147.py", "models/model147/make_bag.py",
    "models/model152/run_model152.py", "models/model152/holdout_components.py", "models/model152/offline_holdout.py",
    "models/model152/run_full.py", "models/model152/bootstrap152.py",
    "models/model154/prepare_holdout.py", "models/model154/run_all.py",
    "models/model156/bayes_evidence.py", "models/model157/oracle_bayes.py", "models/model157/run_model157.py",
    "models/model157/weights_eval157.py", "models/model155/build_splice_union.py",
    "models/model158/run_model158.py", "models/model159/prep_soft.py",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(f"{DEST.name} already exists; refusing to overwrite it")
    seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in seal["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission31 file: {name}")
    constants = json.loads(CONSTANTS.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    import sys
    sys.path.insert(0, str(SERVE))
    from split_tests_numpy import NAMES as SPLIT_NAMES
    stage = Path(tempfile.mkdtemp(prefix=f"{DEST.name}-staging-", dir=ROOT / "submissions"))
    for name in seal["files"]:
        if name in ("main.py", "readme.txt"):
            continue
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    manifest = json.loads((stage / "resources/manifest.json").read_text())
    voters = []
    for arm, fold in ((a, f) for a in ARMS for f in range(5)):
        relative = f"model147_student/{arm}_fold_{fold}.txt"
        destination = stage / "resources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNS / f"{arm}_fold{fold}.txt", destination)
        manifest["file_sha256"][relative] = sha(destination)
        columns = np.load(RUNS / f"{arm}_fold{fold}.npz")["columns"]
        every = list(protocol["contract"]["names"][fold]) + list(SPLIT_NAMES)
        voters.append({"arm": arm, "fold": fold, "path": relative, "names": [every[int(c)] for c in columns]})
    manifest["student_candidate"] = "model147_recut_split_student"
    manifest["model147_student"] = {
        "arm": "+".join(ARMS), "voters": voters, "mu_dist": constants["mu_dist"], "sd_dist": constants["sd_dist"],
        "weight": float(os.environ.get("WSTUDENT", "0.5")),
        "note": f"{DEST.name}: Submission33's student recipe (re-cuts + split tests + distilled targets) refitted on "
                "all labelled series (Model152/154); same top-level z-score blend and constants as Submission33",
    }
    (stage / "resources/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("solution147.py", "split_tests_numpy.py", "main.py"):
        shutil.copy2(SERVE / name, stage / name)
    code = stage / "training_code"
    for relative in TRAINING_CODE:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(HERE / "TRAINING_README.txt", code / "README.txt")
    (stage / "readme.txt").write_text(
        f"{DEST.name}: Submission31 (Submission28 chain + GRU ensemble) with the student of Submission33 (re-cuts of "
        "the training series, 27 causal split-test features, distilled targets) refitted on all labelled training "
        "series. Inference only; the models were trained offline with the code in training_code/ (see "
        "training_code/README.txt); forceFirstTrain=false; trainFrequency=0; INFER_PARALLELISM=12.\n")
    stage.rename(DEST)
    print(f"Built submissions/{DEST.name} with {len(voters)} student voters ({'+'.join(ARMS)}) and "
          f"{len(TRAINING_CODE)} training-code files")


if __name__ == "__main__":
    main()


