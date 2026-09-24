"""Pilot a pre-stack blend of Model84 and Model90 scale probabilities."""
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
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model26 import run_model26 as voter
from models.model39 import run_model39 as incumbent
from models.model67 import run_model67 as source67
from models.model84 import run_model84 as source84
from models.model89 import run_model89 as bayes_source
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT84 = ROOT / CONFIG["model84_run"]
PARENT90 = ROOT / CONFIG["model90_run"]


def initialize():
    if (HERE / "runs/latest.json").exists():
        latest = json.loads((HERE / "runs/latest.json").read_text())
        directory = ROOT / latest["directory"]
        protocol = json.loads((directory / "protocol.json").read_text())
        fingerprint = sha256(directory / "protocol.json")
        if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
            raise ValueError("Changed Model91 protocol")
        for relative, expected in protocol["contract"]["hashes"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Changed frozen Model91 input: {relative}")
        return directory, protocol, fingerprint
    d67, _, s67 = source67.frozen_study()
    d84, _, s84 = source84.frozen_study()
    if (d67 != ROOT / CONFIG["model67_run"] or s67 != CONFIG["model67_protocol_sha256"] or
            d84 != PARENT84 or s84 != CONFIG["model84_protocol_sha256"] or
            sha256(PARENT90 / "protocol.json") != CONFIG["model90_protocol_sha256"]):
        raise ValueError("Changed Model91 parent")
    paths = [
        HERE / "run_model91.py", HERE / "prespec.json", HERE / "readme.txt",
        d67 / "protocol.json", d67 / "summary.json", d84 / "protocol.json",
        d84 / "summary.json", d84 / "pilot_decision.json", PARENT90 / "protocol.json",
        PARENT90 / "pilot_decision.json", PARENT90 / "scale_fold0.npz",
        PARENT90 / "scale_fold0.json", PARENT90 / "scale_fold1.npz",
        PARENT90 / "scale_fold1.json", ROOT / "models/model89/run_model89.py",
        ROOT / "models/model91/run_model91.py", ROOT / "models/model67/run_model67.py",
        ROOT / "models/model84/run_model84.py", ROOT / "models/model20/report.py",
        ROOT / "models/model10/causal_transforms.py", ROOT / "models/model38/run_model38.py",
        ROOT / "train/y_train_index.parquet", ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "tsauc.py",
    ]
    protocol = {"contract": {"config": CONFIG,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths}}}
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", protocol)
    fingerprint = sha256(directory / "protocol.json")
    write_json(HERE / "runs/latest.json", {
        "directory": str(directory.relative_to(ROOT)), "protocol_sha256": fingerprint})
    return directory, protocol, fingerprint


def fold_scores(layout, value):
    return np.asarray([tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)])


def probabilities(layout):
    parents67 = source67.parent_state()
    _, base67, _ = source67.predictions(layout, parents67)
    d84, _, s84 = source84.frozen_study()
    decision84 = json.loads((d84 / "pilot_decision.json").read_text())
    if decision84["selected_arm"] != "expressive" or decision84["selected_weight"] != CONFIG["outer_weight"]:
        raise ValueError("Changed Model84 topology")
    d26, _, s26 = voter.frozen_study()
    original = np.empty(layout.row_count, dtype=np.float32)
    old = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        original[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        saved = read_checkpoint(d84 / f"expressive_fold{fold}.npz", {
            "protocol_sha256": s84, "fold": fold, "arm": "expressive"})
        if saved is None:
            raise ValueError(f"Missing Model84 fold {fold}")
        validate_predictions(saved, held)
        old[held] = saved["prediction"]
    scale = old.copy()
    for fold in CONFIG["pilot_folds"]:
        held = np.flatnonzero(layout.row_fold == fold)
        saved = read_checkpoint(PARENT90 / f"scale_fold{fold}.npz", {
            "protocol_sha256": CONFIG["model90_protocol_sha256"],
            "fold": fold, "arm": "scale"})
        if saved is None:
            raise ValueError(f"Missing Model90 scale fold {fold}")
        validate_predictions(saved, held)
        scale[held] = saved["prediction"]
    m26, m35, _ = incumbent.margins(layout)
    anti_delta = m35.astype(np.float64) - m26.astype(np.float64)
    old_signal = bayes_source.process_probability(old, layout, anti_delta)
    baseline = ((1.0 - CONFIG["outer_weight"]) * np.asarray(base67, dtype=np.float32)
                + CONFIG["outer_weight"] * old_signal).astype(np.float32)
    return np.asarray(base67, dtype=np.float32), old, scale, anti_delta, baseline


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, _, fingerprint = initialize()
        layout = load_development_layout(ROOT)
        base67, old, scale, anti_delta, baseline = probabilities(layout)
        expected = np.asarray(json.loads((PARENT84 / "summary.json").read_text())[
            "candidate_fold_scores"])
        base_scores = fold_scores(layout, baseline)
        np.testing.assert_allclose(base_scores, expected, rtol=0.0, atol=1e-12)
        pilot = np.asarray(CONFIG["pilot_folds"])
        gains, candidate_scores = [], []
        for weight in CONFIG["component_weights"]:
            mixed = ((1.0 - weight) * old + weight * scale).astype(np.float32)
            signal = bayes_source.process_probability(mixed, layout, anti_delta)
            candidate = ((1.0 - CONFIG["outer_weight"]) * base67
                         + CONFIG["outer_weight"] * signal).astype(np.float32)
            scores = fold_scores(layout, candidate)
            gains.append(float(scores[pilot].mean() - base_scores[pilot].mean()))
            candidate_scores.append(scores[pilot].tolist())
            print(f"Model91 component weight {weight:.3f}: pilot gain={gains[-1]:+.9f}", flush=True)
        selected = int(np.argmax(gains))
        decision = {
            "protocol_sha256": fingerprint, "weights": CONFIG["component_weights"],
            "pilot_gains": gains, "candidate_pilot_scores": candidate_scores,
            "baseline_pilot_scores": base_scores[pilot].tolist(),
            "selected_index": selected,
            "selected_weight": CONFIG["component_weights"][selected],
            "continue_confirmation": bool(gains[selected] >= CONFIG["pilot_minimum_gain"]),
        }
        write_json(directory / "pilot_decision.json", decision)
        result = ("# Model91: pre-stack scale-component blend\n\n"
            f"Pilot gains: {dict(zip(CONFIG['component_weights'], gains))}.\n\n"
            f"Selected weight: {decision['selected_weight']}; continue confirmation: "
            f"{decision['continue_confirmation']}.\n\nNo cloud activity.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
