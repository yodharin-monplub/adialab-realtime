"""Pilot native-score addition of maximum-over-tau scale GLR evidence."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest
from models.model91.run_model91 import fold_scores, probabilities

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT84 = ROOT / CONFIG["model84_run"]


def initialize():
    if (HERE / "runs/latest.json").exists():
        latest = json.loads((HERE / "runs/latest.json").read_text())
        directory = ROOT / latest["directory"]
        protocol = json.loads((directory / "protocol.json").read_text())
        fingerprint = sha256(directory / "protocol.json")
        if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
            raise ValueError("Changed Model95 protocol")
        for relative, expected in protocol["contract"]["hashes"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Changed frozen Model95 input: {relative}")
        return directory, fingerprint
    if sha256(PARENT84 / "protocol.json") != CONFIG["model84_protocol_sha256"]:
        raise ValueError("Changed Model95 baseline parent")
    audit = json.loads((HERE / "glr_audit.json").read_text())
    if not audit["label_free"] or audit["rows"] != 4195720:
        raise ValueError("Invalid Model95 label-free audit")
    paths = [HERE / "run_model95.py", HERE / "glr_scale.py", HERE / "build_glr.py",
        HERE / "prespec.json", HERE / "readme.txt", HERE / "real_glr.npy",
        HERE / "glr_audit.json", PARENT84 / "protocol.json", PARENT84 / "summary.json",
        ROOT / "models/model91/run_model91.py", ROOT / "models/model89/run_model89.py",
        ROOT / "models/model10/sources.py", ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet", ROOT / "tsauc.py"]
    protocol = {"contract": {"config": CONFIG,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths}}}
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", protocol)
    fingerprint = sha256(directory / "protocol.json")
    write_json(HERE / "runs/latest.json", {
        "directory": str(directory.relative_to(ROOT)), "protocol_sha256": fingerprint})
    return directory, fingerprint


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, fingerprint = initialize()
        layout = load_development_layout(ROOT)
        _, _, _, _, baseline = probabilities(layout)
        expected = np.asarray(json.loads((PARENT84 / "summary.json").read_text())[
            "candidate_fold_scores"])
        base_scores = fold_scores(layout, baseline)
        np.testing.assert_allclose(base_scores, expected, rtol=0.0, atol=1e-12)
        evidence = np.load(HERE / "real_glr.npy", mmap_mode="r", allow_pickle=False)[:, 0]
        if evidence.shape != (layout.row_count,) or not np.isfinite(evidence).all():
            raise ValueError("Invalid Model95 GLR cache")
        pilot = np.asarray(CONFIG["pilot_folds"])
        gains, scores_by_weight = [], []
        for weight in CONFIG["weights"]:
            candidate = (baseline.astype(np.float64) + weight * evidence).astype(np.float32)
            scores = fold_scores(layout, candidate)
            gains.append(float(scores[pilot].mean() - base_scores[pilot].mean()))
            scores_by_weight.append(scores[pilot].tolist())
            print(f"Model95 weight {weight:.5f}: pilot gain={gains[-1]:+.9f}", flush=True)
        selected = int(np.argmax(gains))
        decision = {"protocol_sha256": fingerprint, "weights": CONFIG["weights"],
            "pilot_gains": gains, "candidate_pilot_scores": scores_by_weight,
            "baseline_pilot_scores": base_scores[pilot].tolist(),
            "selected_index": selected, "selected_weight": CONFIG["weights"][selected],
            "continue_confirmation": bool(gains[selected] >= CONFIG["pilot_minimum_gain"])}
        write_json(directory / "pilot_decision.json", decision)
        result = ("# Model95: maximum-over-tau scale GLR\n\n"
            f"Pilot gains: {dict(zip(CONFIG['weights'], gains))}.\n\n"
            f"Selected weight: {decision['selected_weight']}; continue confirmation: "
            f"{decision['continue_confirmation']}.\n\nNo cloud activity.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
