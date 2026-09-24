"""Fit LightGBM on every original detector plus whitened causal features."""
from __future__ import annotations

import argparse
import fcntl
import gc
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
from models.model20.features import build_cache, load_real_pool, selected_features
from models.model20.run_model20 import synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model44.run_model44 import fit_one
from models.model57 import run_model57 as white_source
from models.model57.whitening import FEATURE_NAMES
from models.model62 import run_model62 as source

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]
PARENT62 = ROOT / CONFIG["model62_run"]
ARM = "lightgbm_aligned_40"


def parent_state():
    for key in ("model20_run", "model20_protocol_sha256", "model57_run",
                "model57_protocol_sha256", "model59_run", "model59_protocol_sha256",
                "model26_protocol_sha256", "model39_protocol_sha256"):
        if CONFIG[key] != source.CONFIG[key]:
            raise ValueError(f"Changed Model62 parent binding: {key}")
    d62, _, s62 = source.frozen_study()
    if d62 != PARENT62 or s62 != CONFIG["model62_protocol_sha256"]:
        raise ValueError("Changed Model62 protocol")
    parent, _, dependencies, d26, d39 = source.parent_state()
    if (CONFIG["lightgbm"] != source.CONFIG["lightgbm"] or
            CONFIG["synthetic_mass"] != source.CONFIG["synthetic_mass"] or
            CONFIG["feature_workers"] != 4):
        raise ValueError("Model64 must change only the feature set")
    if len(parent["contract"]["union_names"]) != 439 or len(FEATURE_NAMES) != 61:
        raise ValueError("Changed original or whitened feature banks")
    bundle_paths = []
    for relative, expected in parent["contract"]["hashes"].items():
        if relative.startswith("submissions/submission16/") and relative.endswith(".py"):
            path = ROOT / relative
            if sha256(path) != expected:
                raise ValueError(f"Changed frozen causal feature source: {relative}")
            bundle_paths.append(path)
    if not bundle_paths:
        raise ValueError("Missing frozen detector source hashes")
    return parent, dependencies, d26, d39, bundle_paths


def initialize(parent, dependencies, d26, d39, bundle_paths):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    base_union = parent["contract"]["union_names"]
    white_names = list(FEATURE_NAMES)
    if set(base_union).intersection(white_names):
        raise ValueError("Original and whitened names overlap")
    missing_names, fold_names = [], []
    for original in parent["contract"]["names"]:
        missing = [name for name in base_union if name not in set(original)]
        names = list(original) + missing + white_names
        if len(original) != 256 or len(missing) != 183 or len(set(names)) != 500:
            raise ValueError("Unexpected original feature selection")
        missing_names.append(missing)
        fold_names.append(names)
    paths = [HERE / "run_model64.py", HERE / "report64.py", HERE / "prespec.json",
        HERE / "readme.txt", PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        PARENT62 / "protocol.json", PARENT62 / "summary.json",
        d26 / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model44/run_model44.py", ROOT / "models/model20/features.py",
        ROOT / "models/model20/run_model20.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model62/run_model62.py", ROOT / "models/model20/report.py",
        ROOT / "models/model21/run_model21.py", ROOT / "models/model10/sources.py",
        ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"] + bundle_paths
    for fold in range(5):
        paths.extend((PARENT20 / f"aligned_25_fold{fold}_features.cache.json",
            PARENT57 / f"aligned_25_fold{fold}_white.json",
            PARENT62 / f"{ARM}_fold{fold}.npz",
            PARENT62 / f"{ARM}_fold{fold}.json"))
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"],
        names=fold_names, missing_names=missing_names,
        union_names=base_union + white_names, extra_names=white_names)
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model64 protocol mismatch")
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
        raise ValueError("Changed Model64 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model64 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model64 input: {relative}")
    return directory, protocol, fingerprint


def prefix_parity(tasks, names):
    for history, online, _ in tasks[:2]:
        full = selected_features(history, online, names)
        for cut in (16, 64, 128):
            if cut <= len(online):
                np.testing.assert_allclose(selected_features(history, online[:cut], names)[-1],
                    full[cut - 1], rtol=1e-5, atol=1e-5)


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model64 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != 1000 or len(sy) != 400_000:
        raise ValueError("Changed Model20 synthetic task count")
    base = read_cache(PARENT20, f"aligned_25_fold{fold}_features", dict(
        protocol_sha256=CONFIG["model20_protocol_sha256"], rows=len(sy),
        names=parent["contract"]["names"][fold], tasks=len(tasks)))
    white = white_source.cached_white(PARENT57 / f"aligned_25_fold{fold}_white.npy",
        CONFIG["model57_protocol_sha256"], len(sy))
    if white is None:
        raise ValueError("Missing Model57 synthetic whitening cache")
    missing_names = protocol["contract"]["missing_names"][fold]
    prefix_parity(tasks, missing_names)
    missing = build_cache(directory, f"aligned_40_fold{fold}_missing", tasks,
        len(sy), missing_names, fingerprint, workers=CONFIG["feature_workers"])
    sx = np.concatenate((np.asarray(base), np.asarray(missing), np.asarray(white)), axis=1)
    if sx.shape != (len(sy), 500):
        raise ValueError("Model64 synthetic feature alignment failed")
    del tasks, base, missing, white
    fit_one(directory, protocol, fingerprint, layout, real, plan, "aligned_40", (sx, sy, st))
    del sx, sy, st
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent, dependencies, d26, d39, bundle_paths = parent_state()
        directory, protocol, fingerprint = initialize(parent, dependencies, d26, d39, bundle_paths)
        layout = load_development_layout(ROOT)
        from models.model64.report64 import report
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
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
