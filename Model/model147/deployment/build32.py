"""Build Submission33 (file name kept from the first draft) = sealed Submission31 + the Model147 student (re-cut data + split-test features).

  ARM147 (env, default SRE40) selects the fold voters models/model147/runs/{ARM}_fold{k}.txt/.npz.
  Blend constants come from models/model112/constants_m147.json (mu_dist / sd_dist of that arm's OOF
  logit, select_blend.py); the student weight stays Submission31's 0.5, as in the holdout evaluation.
  The old distilled voters stay in the bundle (the base classes validate them) but are not used.
  training_code/ carries the TRAINING CODE the competition rules require for prize eligibility
  ("If you submit a pre-trained models without the training code, your model will be ineligible").
Never uploads or submits.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARMS = os.environ.get("ARM147", "SRE40").split(",")          # several arms = a bag of all their fold voters
ARM = os.environ.get("TAG147", "+".join(ARMS))                 # must match the tag of the constants file
SOURCE = ROOT / "submissions/submission31"
DEST = ROOT / "submissions/submission33"   # submission32 already exists (unsent GRU-variant bundle of 2026-09-17)
RUNS = ROOT / "models/model147/runs"
CONSTANTS = ROOT / "models/model112" / os.environ.get("CONST147", "constants_m147.json")
PROTOCOL = ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json"
# sources that reproduce the models in the bundle, in the order they are run (see training_code/README.txt)
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
    "models/model147/holdout_eval147.py", "models/model147/bootstrap_holdout.py",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError("Submission33 already exists; refusing to overwrite it")
    seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in seal["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission31 file: {name}")
    constants = json.loads(CONSTANTS.read_text())
    if constants["tag"] != ARM:
        raise ValueError(f"constants_m147.json was computed for {constants['tag']}, not {ARM}")
    protocol = json.loads(PROTOCOL.read_text())
    import sys
    sys.path.insert(0, str(HERE))
    from split_tests_numpy import NAMES as SPLIT_NAMES
    stage = Path(tempfile.mkdtemp(prefix="submission33-staging-", dir=ROOT / "submissions"))
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
        columns = __import__("numpy").load(RUNS / f"{arm}_fold{fold}.npz")["columns"]
        every = list(protocol["contract"]["names"][fold]) + list(SPLIT_NAMES)
        voters.append({"arm": arm, "fold": fold, "path": relative, "names": [every[int(c)] for c in columns]})
    manifest["student_candidate"] = "model147_recut_split_student"
    manifest["model147_student"] = {
        "arm": ARM, "voters": voters, "mu_dist": constants["mu_dist"], "sd_dist": constants["sd_dist"], "weight": 0.5,
        "note": "Submission33: Submission31's distilled student replaced by the Model147 student "
                "(move-the-start re-cuts + whitened split-test features); same top-level z-score blend",
    }
    (stage / "resources/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(HERE / "solution147.py", stage / "solution147.py")
    shutil.copy2(HERE / "split_tests_numpy.py", stage / "split_tests_numpy.py")
    shutil.copy2(HERE / "main.py", stage / "main.py")
    code = stage / "training_code"
    for relative in TRAINING_CODE:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(HERE / "TRAINING_README.txt", code / "README.txt")
    (stage / "readme.txt").write_text(
        "Submission 33: Submission31 (Submission28 chain + distilled LightGBM student + GRU ensemble) with the "
        "student retrained on move-the-start re-cuts of the training series and 27 causal split-test features "
        "(two-sample tests at an estimated break, AR-whitened; pure-numpy serving). Inference only; the models "
        "were trained offline with the code in training_code/ (see training_code/README.txt); "
        "forceFirstTrain=false; trainFrequency=0; INFER_PARALLELISM=12.\n")
    stage.rename(DEST)
    print(f"Built submissions/submission33 with the {ARM} voters and {len(TRAINING_CODE)} training-code files")


if __name__ == "__main__":
    main()
