"""Build Submission32 = sealed Submission30 + 4-config GRU ensemble (20 numpy GRUs, 2 input contracts).

Requires models/model124/constants_gru32.json (select_gru32.py).  Never uploads or submits.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "submissions/submission30"
DEST = ROOT / "submissions/submission32"
CONSTANTS = ROOT / "models/model124/constants_gru32.json"
CONTRACTS = [ROOT / "models/model124/data/input_contract.json", ROOT / "models/model124/data/sequences_d0_contract.json"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    if DEST.exists():
        raise FileExistsError("Submission32 already exists; refusing to overwrite it")
    seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in seal["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission30 file: {name}")
    constants = json.loads(CONSTANTS.read_text())
    contracts = [json.loads(p.read_text()) for p in CONTRACTS]
    stage = Path(tempfile.mkdtemp(prefix="submission32-staging-", dir=ROOT / "submissions"))
    for name in seal["files"]:
        if name in ("main.py", "readme.txt"):
            continue
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    manifest = json.loads((stage / "resources/manifest.json").read_text())
    folds = []
    for i, cfg in enumerate(constants["configs"]):
        for fold in range(5):
            source = ROOT / cfg["weights"].format(k=fold)
            relative = f"model124_gru/cfg{i}_fold_{fold}.npz"
            destination = stage / "resources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest["file_sha256"][relative] = sha(destination)
            folds.append({"config": i, "fold": fold, "layers": cfg["layers"], "contract": cfg["contract"], "path": relative})
    manifest["gru_candidate"] = "model124_gru_top_blend"
    manifest["model124_gru"] = {
        "folds": folds,
        "input_contracts": [{"names": c["names"], "mu": c["mu"], "sd": c["sd"], "clip": c["clip"]} for c in contracts],
        "mu_gru": constants["mu_gru"], "sd_gru": constants["sd_gru"], "weight_gru": constants["weight_gru"],
        "note": "Submission32: Submission30 + GRU student ensemble (3 x 102-input configs + whitening-only config, 5 folds each), distilled targets, top-level z-score blend",
    }
    write_json(stage / "resources/manifest.json", manifest)
    shutil.copy2(HERE / "solution124.py", stage / "solution124.py")
    shutil.copy2(ROOT / "models/model124/gru_numpy.py", stage / "gru_numpy.py")
    shutil.copy2(HERE / "main.py", stage / "main.py")
    (stage / "readme.txt").write_text(
        "Submission 32: Submission30 (Submission28 chain + oracle-distilled LightGBM top blend) plus a causal "
        "GRU student ensemble (Model124/127: 4 configs x 5 folds = 20 numpy GRUs, two input contracts, "
        "distilled targets) as a second top-level z-score component. Inference only; forceFirstTrain=false; "
        "trainFrequency=0; INFER_PARALLELISM=12.\n")
    stage.rename(DEST)
    print("Built submissions/submission32 with", len(folds), "GRUs")


if __name__ == "__main__":
    main()
