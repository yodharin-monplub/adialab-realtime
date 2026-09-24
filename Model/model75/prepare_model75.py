"""Freeze Model75 inputs and prepare compact labels for remote GPU training."""
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
from models.model18.run_model18 import digest, save_arrays
from models.model20.features import load_real_pool
from models.model20.run_model20 import synthetic_tasks

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def verify_parent(path_key, sha_key):
    directory = ROOT / CONFIG[path_key]
    observed = sha256(directory / "protocol.json")
    if observed != CONFIG[sha_key]:
        raise ValueError(f"Changed {path_key} protocol")
    return directory, json.loads((directory / "protocol.json").read_text())


def source_paths(parent20, parent57, parent64, parent67):
    paths = [
        HERE / "prespec.json",
        HERE / "readme.txt",
        HERE / "prepare_model75.py",
        HERE / "train_model75_gpu.py",
        HERE / "report75.py",
        parent20 / "protocol.json",
        parent20 / "real_features.cache.json",
        parent57 / "protocol.json",
        parent57 / "real_white.json",
        parent64 / "protocol.json",
        parent67 / "protocol.json",
        parent67 / "summary.json",
        ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "models/artifacts/core_features.metadata.npz",
        ROOT / "train/X_train.parquet",
        ROOT / "train/y_train_index.parquet",
        ROOT / "tsauc.py",
    ]
    for fold in range(5):
        paths.extend([
            parent20 / f"aligned_25_fold{fold}_features.cache.json",
            parent57 / f"aligned_25_fold{fold}_white.json",
            parent64 / f"aligned_40_fold{fold}_missing.cache.json",
        ])
    return paths


def initialize():
    parent20, protocol20 = verify_parent("model20_run", "model20_protocol_sha256")
    parent57, _ = verify_parent("model57_run", "model57_protocol_sha256")
    parent64, protocol64 = verify_parent("model64_run", "model64_protocol_sha256")
    parent67, _ = verify_parent("model67_run", "model67_protocol_sha256")
    if protocol20["plans_sha256"] != protocol64["plans_sha256"]:
        raise ValueError("Model20/64 donor plans differ")
    base_names = protocol20["contract"]["union_names"]
    union_names = protocol64["contract"]["union_names"]
    fold_names = protocol64["contract"]["names"]
    if len(base_names) != 439 or len(union_names) != 500:
        raise ValueError("Unexpected Model75 feature count")
    if any(len(names) != 500 or set(names) != set(union_names) for names in fold_names):
        raise ValueError("Invalid fold-specific Model75 feature order")
    paths = source_paths(parent20, parent57, parent64, parent67)
    contract = {
        "config": CONFIG,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "plans_sha256": protocol20["plans_sha256"],
        "base_names": base_names,
        "union_names": union_names,
        "fold_names": fold_names,
    }
    protocol = {"contract": contract}
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model75 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", {
        "directory": str(directory.relative_to(ROOT)),
        "protocol_sha256": fingerprint,
    })
    return directory, protocol, fingerprint, parent20


def validate_frozen(directory, protocol, fingerprint):
    if sha256(directory / "protocol.json") != fingerprint:
        raise ValueError("Changed Model75 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed Model75 input: {relative}")


def prepare_layout(directory, fingerprint, layout):
    path = directory / "layout.npz"
    if not path.exists():
        save_arrays(path,
            target=layout.target.astype(np.int8),
            time_online=layout.time_online.astype(np.int16),
            row_fold=layout.row_fold.astype(np.int8),
            series_position=layout.series_position.astype(np.int32),
            inner_series_fold=layout.inner_series_fold.astype(np.int8),
            series_ids=layout.series_ids.astype(np.int64))
    with np.load(path, allow_pickle=False) as saved:
        if saved["target"].shape != (layout.row_count,):
            raise ValueError("Invalid prepared Model75 layout")
    return {"path": path.name, "sha256": sha256(path), "rows": layout.row_count}


def prepare_synthetic(directory, fingerprint, layout, parent20):
    parent_protocol = json.loads((parent20 / "protocol.json").read_text())
    pool = load_real_pool(layout)
    records = []
    for fold, plan in enumerate(parent_protocol["plans"]):
        path = directory / f"synthetic_fold{fold}.npz"
        if not path.exists():
            tasks, target, time_online = synthetic_tasks(pool, plan, layout, True)
            if len(tasks) != 1000 or len(target) != 400_000:
                raise ValueError("Changed aligned synthetic task count")
            save_arrays(path, target=target.astype(np.int8),
                time_online=time_online.astype(np.int16))
        with np.load(path, allow_pickle=False) as saved:
            if saved["target"].shape != (400_000,) or saved["time_online"].shape != (400_000,):
                raise ValueError(f"Invalid prepared synthetic fold {fold}")
        records.append({"fold": fold, "path": path.name, "sha256": sha256(path), "rows": 400_000})
    return records


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, protocol, fingerprint, parent20 = initialize()
        validate_frozen(directory, protocol, fingerprint)
        layout = load_development_layout(ROOT)
        record = {
            "protocol_sha256": fingerprint,
            "layout": prepare_layout(directory, fingerprint, layout),
            "synthetic": prepare_synthetic(directory, fingerprint, layout, parent20),
        }
        path = directory / "prepared.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("Changed Model75 prepared manifest")
        write_json(path, record)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
