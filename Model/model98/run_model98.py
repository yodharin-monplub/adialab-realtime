"""Pilot residual integrated Bayes evidence on the locked scale GLR."""
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
PARENT90 = ROOT / CONFIG["model90_run"]
PARENT95 = ROOT / CONFIG["model95_run"]


def initialize():
    if (HERE / "runs/latest.json").exists():
        latest = json.loads((HERE / "runs/latest.json").read_text())
        directory = ROOT / latest["directory"]
        protocol = json.loads((directory / "protocol.json").read_text())
        fingerprint = sha256(directory / "protocol.json")
        if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
            raise ValueError("Changed Model98 protocol")
        for relative, expected in protocol["contract"]["hashes"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Changed frozen Model98 input: {relative}")
        return directory, fingerprint
    if (sha256(PARENT84 / "protocol.json") != CONFIG["model84_protocol_sha256"] or
            sha256(PARENT90 / "protocol.json") != CONFIG["model90_protocol_sha256"] or
            sha256(PARENT95 / "protocol.json") != CONFIG["model95_protocol_sha256"]):
        raise ValueError("Changed Model98 parent")
    decision95 = json.loads((PARENT95 / "pilot_decision.json").read_text())
    if (not decision95["continue_confirmation"] or
            decision95["selected_weight"] != CONFIG["glr_weight"]):
        raise ValueError("Model98 does not match the frozen Model95 GLR selection")
    paths = [HERE / "run_model98.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT84 / "protocol.json", PARENT84 / "summary.json",
        PARENT90 / "protocol.json", PARENT90 / "real_evidence.npy",
        PARENT90 / "real_evidence.json", PARENT95 / "protocol.json",
        PARENT95 / "pilot_decision.json", ROOT / "models/model95/real_glr.npy",
        ROOT / "models/model95/glr_audit.json", ROOT / "models/model91/run_model91.py",
        ROOT / "models/model10/sources.py", ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "train/y_train_index.parquet", ROOT / "tsauc.py"]
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
        glr = np.load(ROOT / "models/model95/real_glr.npy", mmap_mode="r",
                      allow_pickle=False)[:, 0]
        evidence = np.load(PARENT90 / "real_evidence.npy", mmap_mode="r",
                           allow_pickle=False)[:, 0]
        if (glr.shape != (layout.row_count,) or evidence.shape != (layout.row_count,) or
                not np.isfinite(glr).all() or not np.isfinite(evidence).all()):
            raise ValueError("Invalid Model98 evidence")
        locked = (baseline.astype(np.float64) + CONFIG["glr_weight"] * glr).astype(np.float32)
        locked_scores = fold_scores(layout, locked)
        pilot = np.asarray(CONFIG["pilot_folds"])
        gains, scores_by_weight = [], []
        for weight in CONFIG["bayes_weights"]:
            candidate = (locked.astype(np.float64) + weight * evidence).astype(np.float32)
            scores = fold_scores(layout, candidate)
            gains.append(float(scores[pilot].mean() - locked_scores[pilot].mean()))
            scores_by_weight.append(scores[pilot].tolist())
            print(f"Model98 Bayes weight {weight:+.3f}: pilot residual gain={gains[-1]:+.9f}",
                  flush=True)
        selected = int(np.argmax(gains))
        decision = {"protocol_sha256": fingerprint, "glr_weight": CONFIG["glr_weight"],
            "bayes_weights": CONFIG["bayes_weights"], "pilot_residual_gains": gains,
            "candidate_pilot_scores": scores_by_weight,
            "model84_pilot_scores": base_scores[pilot].tolist(),
            "locked_glr_pilot_scores": locked_scores[pilot].tolist(),
            "selected_index": selected,
            "selected_bayes_weight": CONFIG["bayes_weights"][selected],
            "continue_confirmation": bool(gains[selected] >= CONFIG["pilot_minimum_gain"])}
        write_json(directory / "pilot_decision.json", decision)
        result = ("# Model98: residual Bayes evidence over scale GLR\n\n"
            f"Pilot residual gains: {dict(zip(CONFIG['bayes_weights'], gains))}.\n\n"
            f"Selected Bayes weight: {decision['selected_bayes_weight']}; continue confirmation: "
            f"{decision['continue_confirmation']}.\n\nNo cloud activity.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
