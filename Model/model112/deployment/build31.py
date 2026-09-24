"""Build Submission31 = sealed Submission28 + 15 seed-bagged distilled voters (Model122).

Voters: Model111 seed (a050) plus Model122 seeds (a050seed1001, a050seed2002)
for each of the five folds; serving bags all 15 logits, which equals the
per-fold 3-seed bag followed by the 5-fold bag used in development.
Constants come from models/model112/constants31.json (select_blend.py run on
the bag3 OOF files).  Never uploads or submits.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "submissions/submission28"
DEST = ROOT / "submissions/submission31"
CONSTANTS = ROOT / "models/model112/constants31.json"
M84 = ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json"
VOTER_FILES = [("model111", "models/model111/runs/a050_fold{k}.txt"),
               ("seed1001", "models/model122/runs/a050seed1001_fold{k}.txt"),
               ("seed2002", "models/model122/runs/a050seed2002_fold{k}.txt")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    if DEST.exists():
        raise FileExistsError("Submission31 already exists; refusing to overwrite it")
    seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in seal["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission28 file: {name}")
    constants = json.loads(CONSTANTS.read_text())
    names = json.loads(M84.read_text())["contract"]["names"]
    stage = Path(tempfile.mkdtemp(prefix="submission31-staging-", dir=ROOT / "submissions"))
    for name in seal["files"]:
        if name in ("main.py", "readme.txt"):
            continue
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    manifest = json.loads((stage / "resources/manifest.json").read_text())
    voters = []
    for fold in range(5):
        if len(names[fold]) != 317:
            raise ValueError("unexpected Model84 contract")
        for label, pattern in VOTER_FILES:
            source = ROOT / pattern.format(k=fold)
            relative = f"model112_distilled/{label}_fold_{fold}.txt"
            destination = stage / "resources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest["file_sha256"][relative] = sha(destination)
            voters.append({"fold": fold, "seed": label, "path": relative, "names": names[fold]})
    manifest["top_level_candidate"] = "model112_distilled_top_blend"
    manifest["model112_distilled"] = {
        "voters": voters,
        **{key: constants[key] for key in ("mu_chain", "sd_chain", "mu_dist", "sd_dist", "weight")},
        "note": "Submission31: 3-seed x 5-fold bag of oracle-distilled expressive LightGBM (Model111 + Model122), top-level z-score blend",
    }
    write_json(stage / "resources/manifest.json", manifest)
    shutil.copy2(HERE / "solution112.py", stage / "solution112.py")
    shutil.copy2(HERE / "main.py", stage / "main.py")
    (stage / "readme.txt").write_text(
        "Submission 31: Submission28 chain + 15 seed-bagged oracle-distilled LightGBM voters "
        "(Model111 + Model122), top-level z-score blend with frozen constants. Inference only; "
        "forceFirstTrain=false; trainFrequency=0; INFER_PARALLELISM=12.\n")
    stage.rename(DEST)
    print("Built submissions/submission31 with", len(voters), "voters")


if __name__ == "__main__":
    main()
