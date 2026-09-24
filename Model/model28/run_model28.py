"""Pilot-selected, confirmatory blend of Model26 and Model27."""
from pathlib import Path
import argparse
import fcntl
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model26 import run_model26 as incumbent


HERE = Path(__file__).resolve().parent
MODEL27 = ROOT / "models/model27"
ARMS = ("model26", "pilot_80")
CONFIG = dict(
    version=1,
    arms=list(ARMS),
    pilot_fold=0,
    confirmation_folds=[1, 2, 3, 4],
    incumbent_weight=.2,
    new_weight=.8,
    target_tsauc=.64,
    minimum_confirmation_gain=.001,
    minimum_confirmation_wins=3,
    bootstrap_replicates=1000,
    bootstrap_seed=281109,
    confidence=.95,
    interpretation=(
        "Model27 fold 0 is an explicitly disclosed pilot used to choose one fixed 80% new-model "
        "weight. Folds 1-4 are the untouched confirmation set for this blend decision. Training "
        "donor exclusions remain those frozen by Model27. The five-fold result is descriptive only."
    ),
)


def candidate_probability(base, new):
    base = np.asarray(base, dtype=np.float32)
    new = np.asarray(new, dtype=np.float32)
    if base.shape != new.shape or not np.isfinite(base).all() or not np.isfinite(new).all():
        raise ValueError("Invalid candidate probabilities")
    return (np.float32(CONFIG["incumbent_weight"]) * base
            + np.float32(CONFIG["new_weight"]) * new)


def _latest(model_dir):
    latest = json.loads((model_dir / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"]:
        raise ValueError(f"Changed upstream protocol: {model_dir.name}")
    return directory, protocol, fingerprint


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    model27_dir, _, model27_sha = _latest(MODEL27)
    model26_dir, _, model26_sha = incumbent.frozen_study()
    files = [HERE / "run_model28.py", HERE / "report.py", HERE / "readme.txt",
             ROOT / "tests/test_model28.py", model27_dir / "protocol.json",
             model26_dir / "protocol.json"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    protocol = dict(contract=dict(config=CONFIG, hashes=hashes,
        model27_protocol_sha256=model27_sha, model26_protocol_sha256=model26_sha))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model28 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    write_json(HERE / "runs/latest.json", dict(
        directory=str(directory.relative_to(ROOT)), protocol_sha256=sha256(path)))
    return directory, protocol, sha256(path)


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model28 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model28 input: {relative}")
    return directory, protocol, fingerprint


def prediction_checkpoint(arm, fold, held):
    model26_dir, _, model26_sha = incumbent.frozen_study()
    base = incumbent.prediction_checkpoint(model26_dir, "depth_mix", fold, model26_sha, held)
    if arm == "model26":
        return base
    if arm != "pilot_80":
        raise ValueError("Unknown Model28 arm")
    model27_dir, _, model27_sha = _latest(MODEL27)
    saved = read_checkpoint(model27_dir / f"matched_context_fold{fold}.npz",
        dict(protocol_sha256=model27_sha, fold=fold, arm="matched_context"))
    if saved is None:
        raise RuntimeError(f"Model27 fold {fold} is not complete")
    validate_predictions(saved, held)
    return candidate_probability(base, saved["prediction"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("initialize", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize()
        if args.command == "report":
            from models.model28.report import report
            report()


if __name__ == "__main__":
    main()
