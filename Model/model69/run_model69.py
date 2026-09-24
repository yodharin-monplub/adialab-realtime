"""Fit Model62 LightGBM plus 18 causal rank-transition columns."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model20.features import block_digest, load_real_pool
from models.model20.run_model20 import synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model44.run_model44 import fit_one
from models.model57 import run_model57 as white_source
from models.model62 import run_model62 as source
from models.model69.transition_features import (FEATURE_NAMES, STATISTICS, WINDOWS,
    transition_features)

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
    d62, p62, s62 = source.frozen_study()
    if d62 != PARENT62 or s62 != CONFIG["model62_protocol_sha256"]:
        raise ValueError("Changed Model62 protocol")
    parent, _, dependencies, d26, d39 = source.parent_state()
    if (CONFIG["lightgbm"] != source.CONFIG["lightgbm"] or
            CONFIG["synthetic_mass"] != source.CONFIG["synthetic_mass"]):
        raise ValueError("Model69 must change only the feature set")
    if (tuple(CONFIG["statistics"]) != STATISTICS or tuple(CONFIG["windows"]) != WINDOWS or
            len(FEATURE_NAMES) != 18 or any(len(names) != 317 for names in p62["contract"]["names"])):
        raise ValueError("Changed Model69 feature contract")
    return parent, p62, dependencies, d26, d39


def initialize(parent, p62, dependencies, d26, d39):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    extra = list(FEATURE_NAMES)
    if set(p62["contract"]["union_names"]).intersection(extra):
        raise ValueError("Model69 feature name collision")
    paths = [HERE / "run_model69.py", HERE / "report69.py", HERE / "transition_features.py",
        HERE / "prespec.json", HERE / "readme.txt", PARENT20 / "protocol.json",
        PARENT20 / "real_features.cache.json", PARENT57 / "protocol.json",
        PARENT57 / "real_white.json", PARENT62 / "protocol.json",
        PARENT62 / "summary.json", d26 / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model44/run_model44.py", ROOT / "models/model62/run_model62.py",
        ROOT / "models/model62/report62.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model57/whitening.py", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model20/report.py", ROOT / "models/model20/features.py",
        ROOT / "models/model21/run_model21.py", ROOT / "models/model10/sources.py",
        ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    for fold in range(5):
        paths.extend((PARENT20 / f"aligned_25_fold{fold}_features.cache.json",
            PARENT57 / f"aligned_25_fold{fold}_white.json",
            PARENT62 / f"{ARM}_fold{fold}.npz",
            PARENT62 / f"{ARM}_fold{fold}.json"))
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"],
        names=[list(names) + extra for names in p62["contract"]["names"]],
        union_names=p62["contract"]["union_names"] + extra,
        extra_names=extra)
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model69 protocol mismatch")
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
        raise ValueError("Changed Model69 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model69 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model69 input: {relative}")
    return directory, protocol, fingerprint


def prefix_parity(tasks):
    for history, online, _ in tasks[:2]:
        full = transition_features(history, online)
        for cut in (1, 16, 64, 128):
            if cut <= len(online):
                np.testing.assert_array_equal(
                    transition_features(history, online[:cut])[-1], full[cut - 1])


def extra_cache(directory, name, tasks, rows, fingerprint):
    path = directory / f"{name}.npy"
    manifest_path = directory / f"{name}.cache.json"
    contract = dict(protocol_sha256=fingerprint, rows=rows,
        names=list(FEATURE_NAMES), tasks=len(tasks))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract"] != contract or not path.exists():
            raise ValueError("Model69 cache contract mismatch")
        matrix = np.load(path, mmap_mode="r+", allow_pickle=False)
        if matrix.shape != (rows, len(FEATURE_NAMES)) or matrix.dtype != np.float32:
            raise ValueError("Model69 cache shape/dtype mismatch")
        for block in manifest["blocks"]:
            if block_digest(matrix[block["start"]:block["end"]]) != block["sha256"]:
                raise ValueError("Corrupt committed Model69 cache block")
    else:
        if shutil.disk_usage(directory).free < rows * len(FEATURE_NAMES) * 4 + 2 * 1024**3:
            raise RuntimeError("Insufficient disk for Model69 feature cache plus 2 GiB reserve")
        matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
            shape=(rows, len(FEATURE_NAMES)))
        manifest = dict(contract=contract, completed_tasks=0, completed_rows=0, blocks=[])
        write_json(manifest_path, manifest)
    completed, offset = manifest["completed_tasks"], manifest["completed_rows"]
    if completed == len(tasks):
        if offset != rows:
            raise ValueError("Incomplete Model69 cache row accounting")
        return np.load(path, mmap_mode="r", allow_pickle=False)
    block_start = offset
    print(f"{name}: restoring {completed}/{len(tasks)} completed series", flush=True)
    for number, (history, online, keep) in enumerate(tasks[completed:], completed + 1):
        features = transition_features(history, online)
        if keep is not None:
            features = features[keep]
        matrix[offset:offset + len(features)] = features
        offset += len(features)
        if number % 64 == 0 or number == len(tasks):
            matrix.flush()
            manifest["blocks"].append(dict(start=block_start, end=offset,
                sha256=block_digest(matrix[block_start:offset])))
            manifest.update(completed_tasks=number, completed_rows=offset)
            write_json(manifest_path, manifest)
            print(f"{name}: {number}/{len(tasks)} series, {offset:,}/{rows:,} rows", flush=True)
            block_start = offset
    if offset != rows:
        raise ValueError("Model69 feature cache did not fill declared rows")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model69 fold {fold} already complete", flush=True)
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
    prefix_parity(tasks)
    extra = extra_cache(directory, f"aligned_40_fold{fold}_transition", tasks, len(sy), fingerprint)
    sx = np.concatenate((np.asarray(base), np.asarray(white), np.asarray(extra)), axis=1)
    if sx.shape != (len(sy), 335):
        raise ValueError("Model69 synthetic feature alignment failed")
    del tasks, base, white, extra
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
        parent, p62, dependencies, d26, d39 = parent_state()
        directory, protocol, fingerprint = initialize(parent, p62, dependencies, d26, d39)
        layout = load_development_layout(ROOT)
        from models.model69.report69 import report
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        pool = load_real_pool(layout)
        real_tasks = [(history, online, None) for _, (history, online) in sorted(pool.items())]
        prefix_parity(real_tasks)
        real_extra = extra_cache(directory, "real_transition", real_tasks,
            layout.row_count, fingerprint)
        del real_tasks
        base_real = read_cache(PARENT20, "real_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"],
            rows=layout.row_count, names=parent["contract"]["union_names"],
            tasks=int(np.sum(layout.inner_series_fold >= 0))))
        white_real = white_source.cached_white(PARENT57 / "real_white.npy",
            CONFIG["model57_protocol_sha256"], layout.row_count)
        if white_real is None:
            raise ValueError("Missing Model57 real whitening cache")
        real = white_source.CombinedFeatures(
            white_source.CombinedFeatures(base_real, white_real), real_extra)
        for fold in CONFIG["pilot_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
