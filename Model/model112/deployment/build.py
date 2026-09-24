"""Build Submission30 = sealed Submission28 + top-level Model111 distilled voters.

Requires models/model112/constants.json (written by select_blend.py).
Never uploads or submits.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "submissions/submission28"
DEST = ROOT / "submissions/submission30"
CONSTANTS = ROOT / "models/model112/constants.json"
RUN111 = ROOT / "models/model111/runs"
M84 = ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    if DEST.exists():
        raise FileExistsError("Submission30 already exists; refusing to overwrite it")
    seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in seal["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission28 file: {name}")
    constants = json.loads(CONSTANTS.read_text())
    names = json.loads(M84.read_text())["contract"]["names"]
    stage = Path(tempfile.mkdtemp(prefix="submission30-staging-", dir=ROOT / "submissions"))
    for name in seal["files"]:
        if name in ("main.py", "readme.txt"):
            continue
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    manifest = json.loads((stage / "resources/manifest.json").read_text())
    voters = []
    for fold in range(5):
        source = RUN111 / f"{constants['tag']}_fold{fold}.txt"
        relative = f"model112_distilled/fold_{fold}.txt"
        destination = stage / "resources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest["file_sha256"][relative] = sha(destination)
        if len(names[fold]) != 317:
            raise ValueError("unexpected Model84 contract")
        voters.append({"fold": fold, "path": relative, "names": names[fold]})
    manifest["top_level_candidate"] = "model112_distilled_top_blend"
    manifest["model112_distilled"] = {
        "voters": voters,
        **{key: constants[key] for key in ("mu_chain", "sd_chain", "mu_dist", "sd_dist", "weight")},
        "note": "Model111 oracle-distilled expressive LightGBM, top-level z-score blend with the Submission28 chain",
    }
    write_json(stage / "resources/manifest.json", manifest)
    shutil.copy2(HERE / "solution112.py", stage / "solution112.py")
    shutil.copy2(HERE / "main.py", stage / "main.py")
    shutil.copy2(HERE / "readme_submission.txt", stage / "readme.txt")
    stage.rename(DEST)
    print("Built submissions/submission30")


if __name__ == "__main__":
    main()
