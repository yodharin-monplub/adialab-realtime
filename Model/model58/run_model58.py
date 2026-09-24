"""Fit XGBoost using only Model57's frozen whitened innovation features."""
from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import synthetic_tasks, train_one
from models.model26 import run_model26 as voter
from models.model39 import run_model39 as incumbent
from models.model57 import run_model57 as white_source
from models.model57.whitening import FEATURE_NAMES

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]


def parent_state():
    if sha256(PARENT20 / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 protocol")
    directory57, protocol57, sha57 = white_source.frozen_study()
    if directory57 != PARENT57 or sha57 != CONFIG["model57_protocol_sha256"]:
        raise ValueError("Changed Model57 protocol")
    d26, _, s26 = voter.frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed incumbent protocol")
    parent20 = json.loads((PARENT20 / "protocol.json").read_text())
    if digest(parent20["plans"]) != parent20["plans_sha256"]:
        raise ValueError("Changed Model20 donor plan")
    if protocol57["plans"] != parent20["plans"] or protocol57["contract"]["extra_names"] != list(FEATURE_NAMES):
        raise ValueError("Model57 cache is not aligned with Model20 training rows")
    dependencies = {name: importlib.metadata.version(name) for name in
        ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")}
    dependencies.update(xgboost=xgb.__version__,
        xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    if dependencies != parent20["contract"]["dependencies"]:
        raise ValueError("Use Model20 dependency environment")
    return parent20, dependencies, d26, d39


def initialize(parent, dependencies, d26, d39):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model58.py", HERE / "report58.py", HERE / "prespec.json",
        HERE / "readme.txt", PARENT20 / "protocol.json", PARENT57 / "protocol.json",
        PARENT57 / "real_white.json", d26 / "protocol.json",
        d39 / "protocol.json", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model20/report.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model57/whitening.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    paths += [PARENT57 / f"aligned_25_fold{fold}_white.json" for fold in range(5)]
    names = list(FEATURE_NAMES)
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"],
        names=[names for _ in range(5)], union_names=names)
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model58 protocol mismatch")
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
        raise ValueError("Changed Model58 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model58 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model58 input: {relative}")
    return directory, protocol, fingerprint


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"aligned_25_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="aligned_25"))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model58 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != 1000 or len(sy) != 400_000:
        raise ValueError("Changed Model20 synthetic task count")
    sx = white_source.cached_white(PARENT57 / f"aligned_25_fold{fold}_white.npy",
        CONFIG["model57_protocol_sha256"], len(sy))
    if sx is None:
        raise ValueError("Missing Model57 synthetic whitening cache")
    train_one(directory, protocol, fingerprint, layout, real, plan,
        "aligned_25", (sx, sy, st))
    del tasks, sx, sy, st
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent, dependencies, d26, d39 = parent_state()
        directory, protocol, fingerprint = initialize(parent, dependencies, d26, d39)
        layout = load_development_layout(ROOT)
        from models.model58.report58 import report
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        real = white_source.cached_white(PARENT57 / "real_white.npy",
            CONFIG["model57_protocol_sha256"], layout.row_count)
        if real is None:
            raise ValueError("Missing Model57 real whitening cache")
        pool = load_real_pool(layout)
        for fold in CONFIG["pilot_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
