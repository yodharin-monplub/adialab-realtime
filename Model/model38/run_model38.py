"""Frozen confirmation runner for Model38's pilot-selected causal smoother."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.causal_transforms import TransformSpec, apply_causal_transform
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as incumbent


HERE = Path(__file__).resolve().parent
PRESPEC = HERE / "prespec.json"
CONFIG = json.loads(PRESPEC.read_text())


def transform_spec():
    spec = TransformSpec(**CONFIG["transform"])
    spec.validate()
    return spec


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    incumbent_dir, _, incumbent_sha = incumbent.frozen_study()
    if incumbent_sha != CONFIG["model26_protocol_sha256"]:
        raise ValueError("Changed Model26 protocol")
    files = [HERE / "run_model38.py", HERE / "report.py", HERE / "readme.txt", PRESPEC,
             ROOT / "tests/test_model38.py", incumbent_dir / "protocol.json"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        model26_run=str(incumbent_dir.relative_to(ROOT)), model26_protocol_sha256=incumbent_sha))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model38 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", dict(
        directory=str(directory.relative_to(ROOT)), protocol_sha256=fingerprint))
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model38 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model38 input: {relative}")
    return directory, protocol, fingerprint


def margins(layout):
    incumbent_dir, _, incumbent_sha = incumbent.frozen_study()
    probability = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        probability[held] = incumbent.prediction_checkpoint(
            incumbent_dir, "depth_mix", fold, incumbent_sha, held)
    base = ensemble_margin(probability, layout)
    candidate = apply_causal_transform(expit(base.astype(np.float64)),
        layout.series_position, transform_spec(), layout.time_online).astype(np.float32)
    return base, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize()
        if args.command == "report":
            from models.model38.report import report
            report()


if __name__ == "__main__":
    main()

