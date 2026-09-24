"""Build Submission34 = sealed Submission33 + the Model148 GRU ensemble (129 inputs incl. split tests).

Needs models/model148/constants_gru148.json (select_gru148.py), the fold weights
models/model148/{tag}fold{k}_weights.npz and data/input_contract_split.json.  Never uploads.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
M148 = ROOT / "models/model148"
SOURCE = ROOT / "submissions/submission33"
DEST = ROOT / "submissions/submission34"
EXTRA_TRAINING_CODE = ["models/model148/export_split_inputs.py", "models/model148/select_gru148.py",
                       "models/model148/blend_eval148.py", "models/model148/weights_eval.py",
                       "models/model148/holdout_eval_bundle.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError("Submission34 already exists; refusing to overwrite it")
    seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in seal["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission33 file: {name}")
    constants = json.loads((M148 / "constants_gru148.json").read_text())
    contract = json.loads((M148 / "data/input_contract_split.json").read_text())
    stage = Path(tempfile.mkdtemp(prefix="submission34-staging-", dir=ROOT / "submissions"))
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
            relative = f"model148_gru/{cfg['tag']}fold_{fold}.npz"
            destination = stage / "resources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(M148 / f"{cfg['tag']}fold{fold}_weights.npz", destination)
            manifest["file_sha256"][relative] = sha(destination)
            folds.append({"config": cfg["tag"], "fold": fold, "layers": cfg["layers"], "path": relative})
    manifest["gru148_candidate"] = "model148_split_input_gru"
    manifest["model148_gru"] = {
        "folds": folds, "split_names": contract["split_names"], "split_mu": contract["split_mu"],
        "split_sd": contract["split_sd"], "clip": contract["clip"],
        "mu_gru": constants["mu_gru"], "sd_gru": constants["sd_gru"], "weight_gru": constants["weight_gru"],
        "note": "Submission34: GRU ensemble retrained on Submission31's 102 inputs + the 27 split-test columns",
    }
    (stage / "resources/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(HERE / "solution148.py", stage / "solution148.py")
    main_py = (SOURCE / "main.py").read_text().replace("solution147", "solution148").replace(
        "Submission32 entry point (Submission31 + Model147 student)",
        "Submission34 entry point (Submission31 + Model147 student + Model148 GRUs)")
    (stage / "main.py").write_text(main_py)
    for relative in EXTRA_TRAINING_CODE:
        target = stage / "training_code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    readme = stage / "training_code/README.txt"
    readme.write_text(readme.read_text() + "\n11. models/model148/export_split_inputs.py, models/model124/train_gru.py, select_gru148.py\n"
                      "        the GRU ensemble retrained on 129 inputs (the 102 of step 7 + the 27 split-test columns).\n")
    (stage / "readme.txt").write_text(
        "Submission 34: Submission33 (chain + Model147 student on re-cut data and split-test features + GRU ensemble) "
        "with the GRU ensemble retrained on 129 inputs (the 27 causal split-test features added). Inference only; "
        "training code in training_code/; forceFirstTrain=false; trainFrequency=0; INFER_PARALLELISM=12.\n")
    stage.rename(DEST)
    print("Built submissions/submission34 with", len(folds), "GRUs")


if __name__ == "__main__":
    main()
