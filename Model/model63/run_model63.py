"""Fit extra-randomized LightGBM with 40% aligned-synthetic mass."""
from __future__ import annotations

import argparse
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
from models.model20.features import load_real_pool
from models.model21.run_model21 import read_cache
from models.model57 import run_model57 as white_source
from models.model61 import run_model61 as extra_source
from models.model62 import run_model62 as mass_source

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]
PARENT61 = ROOT / CONFIG["model61_run"]
PARENT62 = ROOT / CONFIG["model62_run"]


def parent_state():
    for key in ("model20_run", "model20_protocol_sha256", "model57_run",
                "model57_protocol_sha256", "model59_run", "model59_protocol_sha256",
                "model26_protocol_sha256", "model39_protocol_sha256"):
        if CONFIG[key] != mass_source.CONFIG[key]:
            raise ValueError(f"Changed Model62 parent binding: {key}")
    d62, p62, s62 = mass_source.frozen_study()
    d61, _, s61 = extra_source.frozen_study()
    if d62 != PARENT62 or s62 != CONFIG["model62_protocol_sha256"]:
        raise ValueError("Changed Model62 protocol")
    if d61 != PARENT61 or s61 != CONFIG["model61_protocol_sha256"]:
        raise ValueError("Changed Model61 protocol")
    if (CONFIG["lightgbm"] != extra_source.CONFIG["lightgbm"] or
            CONFIG["synthetic_mass"] != mass_source.CONFIG["synthetic_mass"]):
        raise ValueError("Model63 must combine Model61 recipe with Model62 synthetic mass")
    parent, _, dependencies, d26, d39 = mass_source.parent_state()
    return parent, p62, dependencies, d26, d39


def initialize(parent, p62, dependencies, d26, d39):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model63.py", HERE / "report63.py", HERE / "prespec.json",
        HERE / "readme.txt", PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        PARENT61 / "protocol.json", PARENT61 / "summary.json",
        PARENT62 / "protocol.json", PARENT62 / "summary.json",
        d26 / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model44/run_model44.py", ROOT / "models/model62/run_model62.py",
        ROOT / "models/model61/run_model61.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model20/run_model20.py", ROOT / "models/model20/report.py",
        ROOT / "models/model21/run_model21.py", ROOT / "models/model10/sources.py",
        ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    for fold in range(5):
        paths.extend((PARENT20 / f"aligned_25_fold{fold}_features.cache.json",
            PARENT57 / f"aligned_25_fold{fold}_white.json",
            PARENT62 / f"lightgbm_aligned_40_fold{fold}.npz",
            PARENT62 / f"lightgbm_aligned_40_fold{fold}.json"))
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"],
        names=p62["contract"]["names"],
        union_names=p62["contract"]["union_names"],
        extra_names=p62["contract"]["extra_names"])
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model63 protocol mismatch")
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
        raise ValueError("Changed Model63 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model63 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model63 input: {relative}")
    return directory, protocol, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent, p62, dependencies, d26, d39 = parent_state()
        directory, protocol, fingerprint = initialize(parent, p62, dependencies, d26, d39)
        layout = load_development_layout(ROOT)
        from models.model63.report63 import report
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        base_real = read_cache(PARENT20, "real_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"],
            rows=layout.row_count, names=parent["contract"]["union_names"],
            tasks=int(np.sum(layout.inner_series_fold >= 0))))
        white_real = white_source.cached_white(PARENT57 / "real_white.npy",
            CONFIG["model57_protocol_sha256"], layout.row_count)
        if white_real is None:
            raise ValueError("Missing Model57 real whitening cache")
        real = white_source.CombinedFeatures(base_real, white_real)
        pool = load_real_pool(layout)
        for fold in CONFIG["pilot_folds"]:
            mass_source.run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                mass_source.run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
