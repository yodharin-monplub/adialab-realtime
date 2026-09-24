"""Frozen exploratory anti-residual combination."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as incumbent
from models.model34.run_model34 import prediction_checkpoint as model34_prediction


HERE = Path(__file__).resolve().parent
PRESPEC = HERE / "prespec.json"
CONFIG = json.loads(PRESPEC.read_text())
MODEL30 = ROOT / "models/model30"
MODEL34 = ROOT / "models/model34"
ARMS = ("model26", "anti_residual")


def _state(model_dir, expected):
    latest = json.loads((model_dir / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or fingerprint != expected:
        raise ValueError(f"Changed {model_dir.name} protocol")
    return directory, fingerprint


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d30, s30 = _state(MODEL30, CONFIG["model30_protocol_sha256"])
    d34, s34 = _state(MODEL34, CONFIG["model34_protocol_sha256"])
    d26, _, s26 = incumbent.frozen_study()
    files = [HERE / "run_model35.py", HERE / "report.py", HERE / "readme.txt", PRESPEC,
        ROOT / "tests/test_model35.py", d30 / "protocol.json", d34 / "protocol.json", d26 / "protocol.json"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        model26_protocol_sha256=s26, model30_protocol_sha256=s30, model34_protocol_sha256=s34))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model35 protocol mismatch")
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
        raise ValueError("Changed Model35 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model35 input: {relative}")
    return directory, protocol, fingerprint


def margins(layout):
    d26, _, s26 = incumbent.frozen_study()
    d30, s30 = _state(MODEL30, CONFIG["model30_protocol_sha256"])
    d34, s34 = _state(MODEL34, CONFIG["model34_protocol_sha256"])
    base_probability = np.empty(layout.row_count, dtype=np.float32)
    p30 = np.empty_like(base_probability)
    p34 = np.empty_like(base_probability)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        base_probability[held] = incumbent.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        saved = read_checkpoint(d30 / f"matched_15_fold{fold}.npz",
            dict(protocol_sha256=s30, fold=fold, arm="matched_15"))
        if saved is None:
            raise RuntimeError(f"Model30 fold {fold} incomplete")
        validate_predictions(saved, held)
        p30[held] = saved["prediction"]
        p34[held] = model34_prediction(d34, fold, s34, held)
    base = ensemble_margin(base_probability, layout)
    m34 = logit(np.clip(p34, 1e-7, 1 - 1e-7))
    component_delta = (logit(np.clip(p30, 1e-7, 1 - 1e-7))
                       - logit(np.clip(base_probability, 1e-7, 1 - 1e-7)))
    candidate = (base + CONFIG["model34_residual_coefficient"] * (m34 - base)
                 + CONFIG["model30_component_delta_coefficient"] * component_delta).astype(np.float32)
    if not np.isfinite(candidate).all():
        raise ValueError("Model35 produced non-finite margins")
    return base, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("initialize", "run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize()
        if args.command != "initialize":
            from models.model35.report import report
            report()


if __name__ == "__main__":
    main()
