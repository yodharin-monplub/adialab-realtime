"""Build Submission31 = sealed Submission30 + Model124 GRU ensemble (15 numpy GRUs).

Requires models/model124/constants_gru.json (select_gru.py) and the fold
weights in models/model124/remote_runs/.  Never uploads or submits.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "submissions/submission30"
DEST = ROOT / "submissions/submission31"
CONSTANTS = ROOT / "models/model124/constants_gru.json"
CONTRACT = ROOT / "models/model124/data/input_contract.json"
RUNS = ROOT / "models/model124/remote_runs"


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
            raise ValueError(f"Changed sealed Submission30 file: {name}")
    constants = json.loads(CONSTANTS.read_text())
    contract = json.loads(CONTRACT.read_text())
    stage = Path(tempfile.mkdtemp(prefix="submission31-staging-", dir=ROOT / "submissions"))
    for name in seal["files"]:
        if name in ("main.py", "readme.txt"):
            continue
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    manifest = json.loads((stage / "resources/manifest.json").read_text())
    folds = []
    for cfg in constants["configs"]:
        for fold in range(5):
            source = RUNS / f"{cfg['tag']}fold{fold}_weights.npz"
            relative = f"model124_gru/{cfg['tag'] or 'h192_'}fold_{fold}.npz"
            destination = stage / "resources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest["file_sha256"][relative] = sha(destination)
            folds.append({"config": cfg["tag"] or "h192_", "fold": fold, "layers": cfg["layers"], "path": relative})
    manifest["gru_candidate"] = "model124_gru_top_blend"  # top_level_candidate stays Model112's
    manifest["model124_gru"] = {
        "folds": folds, "input_contract": {"names": contract["names"], "mu": contract["mu"],
                                           "sd": contract["sd"], "clip": contract["clip"]},
        "mu_gru": constants["mu_gru"], "sd_gru": constants["sd_gru"], "weight_gru": constants["weight_gru"],
        "note": "Submission31: Submission30 + causal GRU student ensemble (3 configs x 5 folds), distilled targets, top-level z-score blend",
    }
    write_json(stage / "resources/manifest.json", manifest)
    shutil.copy2(HERE / "solution124.py", stage / "solution124.py")
    shutil.copy2(ROOT / "models/model124/gru_numpy.py", stage / "gru_numpy.py")
    shutil.copy2(HERE / "main.py", stage / "main.py")
    (stage / "readme.txt").write_text(
        "Submission 31: Submission30 (Submission28 chain + oracle-distilled LightGBM top blend) plus a "
        "causal GRU student ensemble (Model124: 3 configs x 5 folds, numpy float32 inference, distilled "
        "targets) as a second top-level z-score component. Inference only; forceFirstTrain=false; "
        "trainFrequency=0; INFER_PARALLELISM=12.\n")
    stage.rename(DEST)
    print("Built submissions/submission31 with", len(folds), "GRUs")


if __name__ == "__main__":
    main()
