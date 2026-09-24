"""Frozen composition of Model35 and Model38's confirmed causal smoother."""
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

from models.model10.causal_transforms import apply_causal_transform
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest
from models.model35 import run_model35 as anti_residual
from models.model38 import run_model38 as smoother


HERE = Path(__file__).resolve().parent
PRESPEC = HERE / "prespec.json"
CONFIG = json.loads(PRESPEC.read_text())


def _state(module, expected):
    directory, protocol, fingerprint = module.frozen_study()
    if fingerprint != expected:
        raise ValueError(f"Changed {module.__name__} protocol")
    return directory, protocol, fingerprint


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d35, _, s35 = _state(anti_residual, CONFIG["model35_protocol_sha256"])
    d38, _, s38 = _state(smoother, CONFIG["model38_protocol_sha256"])
    files = [HERE / "run_model39.py", HERE / "report.py", HERE / "readme.txt", PRESPEC,
             ROOT / "tests/test_model39.py", d35 / "protocol.json", d38 / "protocol.json",
             ROOT / "models/model35/run_model35.py", ROOT / "models/model38/run_model38.py"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        model35_protocol_sha256=s35, model38_protocol_sha256=s38))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model39 protocol mismatch")
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
        raise ValueError("Changed Model39 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model39 input: {relative}")
    return directory, protocol, fingerprint


def margins(layout):
    model26, model35 = anti_residual.margins(layout)
    model39 = apply_causal_transform(expit(model35.astype(np.float64)),
        layout.series_position, smoother.transform_spec(), layout.time_online).astype(np.float32)
    return model26, model35, model39


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize()
        if args.command == "report":
            from models.model39.report import report
            report()


if __name__ == "__main__":
    main()

