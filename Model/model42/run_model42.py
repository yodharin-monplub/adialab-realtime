"""Frozen confirmation of a pilot-selected secondary Model39 smoother."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from models.model10.causal_transforms import TransformSpec, apply_causal_transform
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest
from models.model39 import run_model39 as parent

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def transform_spec():
    value = TransformSpec(**CONFIG["transform"]); value.validate(); return value


def initialize():
    if (HERE / "runs/latest.json").exists(): return frozen_study()
    parent_dir, _, parent_sha = parent.frozen_study()
    if parent_sha != CONFIG["model39_protocol_sha256"]: raise ValueError("Changed Model39 protocol")
    files = [HERE / "run_model42.py", HERE / "report.py", HERE / "readme.txt", HERE / "prespec.json",
             ROOT / "tests/test_model42.py", parent_dir / "protocol.json"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files}))
    directory = HERE / "runs" / digest(protocol)[:16]; directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError("Model42 protocol mismatch")
    if not path.exists(): write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=fingerprint))
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text()); directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text()); fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model42 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected: raise ValueError(f"Changed frozen Model42 input: {relative}")
    return directory, protocol, fingerprint


def margins(layout):
    model26, _, model39 = parent.margins(layout)
    candidate = apply_causal_transform(expit(model39.astype(float)), layout.series_position,
        transform_spec(), layout.time_online).astype(np.float32)
    return model26, model39, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("initialize", "report"))
    args = parser.parse_args(); (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); initialize()
        if args.command == "report":
            from models.model42.report import report
            report()
if __name__ == "__main__": main()

