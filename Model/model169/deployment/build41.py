"""Build a bundle whose GRU part is replaced by a newer set of networks (Submission41: Model169 nets; Submission42: the Model170 all-series nets with weight_gru from the Model171 protocol).
Env: SOURCE (sealed bundle to start from), NETSDIR, GRUTAG, WGRU, DEST.


The deployed GRU part is 15 small nets (h192 / h96x3 / h128) trained on the OLD teacher's targets; it scores
0.61905 on the sealed holdout.  The replacement is 5 nets (hidden 256, 2 layers, one seed x five folds)
trained on the GPU with the Model157 oracle-teacher targets: raw 0.62201, and in the blend (honest dev-only
student, student weight 1.5) 0.638015 vs 0.637246 for the deployed nets.
The bundle's serving code already accepts 5, 10 or 15 GRUs with a layer count each, so only the weight files
and the manifest entry change.  z-score constants are the mean/sd of the new nets' development out-of-fold
logits (label-free, exactly how the deployed ones were computed).
Usage: python models/model169/deployment/build41.py        (env DEST, default submission41)
Never uploads.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "submissions" / os.environ.get("SOURCE", "submission40")
DEST = ROOT / "submissions" / os.environ.get("DEST", "submission41")
NETS = ROOT / os.environ.get("NETSDIR", "models/model169/runs/gpu")
TAG = os.environ.get("GRUTAG", "h256s0_")
LAYERS = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(f"{DEST.name} already exists")
    seal = json.loads((SOURCE / "bundle_manifest.json").read_text())
    for name, record in seal["files"].items():
        if sha(SOURCE / name) != record["sha256"]:
            raise ValueError(f"Changed sealed Submission40 file: {name}")
    stage = Path(tempfile.mkdtemp(prefix=f"{DEST.name}-staging-", dir=ROOT / "submissions"))
    for name in seal["files"]:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    manifest = json.loads((stage / "resources/manifest.json").read_text())
    # drop the old GRU files and their hashes
    for old in list(manifest["file_sha256"]):
        if old.startswith("model124_gru/"):
            manifest["file_sha256"].pop(old)
            (stage / "resources" / old).unlink()
    folds = []
    logits = []
    for k in range(5):
        rel = f"model124_gru/{TAG}fold_{k}.npz"
        shutil.copy2(NETS / f"{TAG}fold{k}_weights.npz", stage / "resources" / rel)
        manifest["file_sha256"][rel] = sha(stage / "resources" / rel)
        folds.append({"config": TAG, "fold": k, "layers": LAYERS, "path": rel})
        p = np.clip(np.load(NETS / f"{TAG}fold{k}_pred.npz")["prediction"].astype(np.float64), 1e-6, 1 - 1e-6)
        logits.append(np.log(p) - np.log1p(-p))
    z = np.concatenate(logits)
    spec = dict(manifest["model124_gru"])
    if os.environ.get("WGRU"):
        spec["weight_gru"] = float(os.environ["WGRU"])
    spec.update(folds=folds, mu_gru=float(z.mean()), sd_gru=float(z.std()),
                note=f"{DEST.name}: 5 GRU students (hidden 256, 2 layers, one seed x five folds) trained on the "
                     "Model157 oracle-teacher soft targets (Model169, rented RTX 3090); constants are the mean/sd "
                     "of their development out-of-fold logits")
    manifest["model124_gru"] = spec
    (stage / "resources/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (stage / "readme.txt").write_text(
        (SOURCE / "readme.txt").read_text().rstrip("\n") +
        " The GRU component is the Model169 set: five hidden-256 networks trained on the oracle-teacher targets.\n")
    code = stage / "training_code"
    for rel in ("models/model169/train_gpu.py", "models/model169/score_gpu.py", "models/model161/soft_b.py",
                "models/model169/vast.py"):
        target = code / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (stage / "bundle_manifest.json").unlink(missing_ok=True)
    stage.rename(DEST)
    print(f"Built submissions/{DEST.name}: 5 GRUs ({TAG}), mu {z.mean():.5f} sd {z.std():.5f}")


if __name__ == "__main__":
    main()
