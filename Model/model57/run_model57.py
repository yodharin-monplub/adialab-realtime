"""History-whitened innovation features atop the Model20 synthetic voter."""
from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

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
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as voter
from models.model39 import run_model39 as incumbent
from models.model57.whitening import AR_ORDERS, EWMA_LAMBDA, FEATURE_NAMES, WINDOWS, whitening_features

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model20_run"]


class CombinedFeatures:
    """Read-only row selector; avoids materializing a 500-column full cache."""
    def __init__(self, base, extra):
        if base.shape[0] != extra.shape[0]:
            raise ValueError("Misaligned Model57 real feature caches")
        self.base, self.extra = base, extra
        self.shape = (base.shape[0], base.shape[1] + extra.shape[1])

    def __getitem__(self, rows):
        return np.concatenate((np.asarray(self.base[rows]),
            np.asarray(self.extra[rows])), axis=1)


def parent_state():
    if sha256(PARENT / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 protocol")
    d26, _, s26 = voter.frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed incumbent protocol")
    parent = json.loads((PARENT / "protocol.json").read_text())
    if digest(parent["plans"]) != parent["plans_sha256"]:
        raise ValueError("Changed Model20 donor plan")
    if (tuple(CONFIG["ar_orders"]) != AR_ORDERS or CONFIG["ewma_lambda"] != EWMA_LAMBDA
            or tuple(CONFIG["windows"]) != WINDOWS or len(FEATURE_NAMES) != 61):
        raise ValueError("Changed whitening feature specification")
    dependencies = {name: importlib.metadata.version(name) for name in
        ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")}
    dependencies.update(xgboost=xgb.__version__,
        xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    if dependencies != parent["contract"]["dependencies"]:
        raise ValueError("Use Model20 dependency environment")
    return parent, dependencies, d26, d39


def initialize(parent, dependencies, d26, d39):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model57.py", HERE / "report57.py", HERE / "whitening.py",
        HERE / "prespec.json", HERE / "readme.txt", PARENT / "protocol.json",
        PARENT / "real_features.cache.json", d26 / "protocol.json",
        d39 / "protocol.json", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model20/report.py", ROOT / "models/model21/run_model21.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "train/X_train.parquet",
        ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    paths += [PARENT / f"aligned_25_fold{fold}_features.cache.json" for fold in range(5)]
    base_union = parent["contract"]["union_names"]
    extra = list(FEATURE_NAMES)
    if set(base_union).intersection(extra):
        raise ValueError("Whitening feature name collision")
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"],
        names=[list(names) + extra for names in parent["contract"]["names"]],
        union_names=base_union + extra, extra_names=extra)
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model57 protocol mismatch")
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
        raise ValueError("Changed Model57 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model57 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model57 input: {relative}")
    return directory, protocol, fingerprint


def cached_white(path, fingerprint, rows):
    manifest_path = path.with_suffix(".json")
    if not manifest_path.exists():
        return None
    record = json.loads(manifest_path.read_text())
    expected = dict(protocol_sha256=fingerprint, rows=rows,
        names=list(FEATURE_NAMES), sha256=sha256(path))
    if record != expected:
        raise ValueError(f"Corrupt Model57 feature cache: {path}")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (rows, len(FEATURE_NAMES)) or matrix.dtype != np.float32:
        raise ValueError(f"Wrong Model57 feature-cache shape: {path}")
    return matrix


def prefix_parity(history, online):
    full = whitening_features(history, online)
    for cut in (1, 16, 64, 128, 256):
        if cut <= len(online):
            np.testing.assert_array_equal(
                whitening_features(history, online[:cut])[-1], full[cut - 1])
    return full


def real_white(directory, fingerprint, layout, pool):
    path = directory / "real_white.npy"
    saved = cached_white(path, fingerprint, layout.row_count)
    if saved is not None:
        return saved
    partial = directory / "real_white.partial.npy"
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
        shape=(layout.row_count, len(FEATURE_NAMES)))
    positions = layout.series_position
    total = 0
    started = time.monotonic()
    for number, (position, (history, online)) in enumerate(sorted(pool.items())):
        left, right = np.searchsorted(positions, (position, position + 1))
        if right - left != len(online):
            raise ValueError("Real whitening rows do not align with series")
        matrix[left:right] = (prefix_parity(history, online) if number < 3 else
                              whitening_features(history, online))
        total += right - left
        if (number + 1) % 500 == 0:
            matrix.flush()
            print(f"Model57 real whitening cache: {number + 1}/{len(pool)} series", flush=True)
    if total != layout.row_count:
        raise ValueError("Model57 real whitening cache is incomplete")
    matrix.flush()
    del matrix
    partial.replace(path)
    write_json(path.with_suffix(".json"), dict(protocol_sha256=fingerprint,
        rows=layout.row_count, names=list(FEATURE_NAMES), sha256=sha256(path)))
    print(f"Model57 real whitening features complete in {time.monotonic()-started:.1f}s", flush=True)
    return cached_white(path, fingerprint, layout.row_count)


def synthetic_white(directory, fingerprint, fold, tasks, rows):
    path = directory / f"aligned_25_fold{fold}_white.npy"
    saved = cached_white(path, fingerprint, rows)
    if saved is not None:
        return saved
    partial = directory / f"aligned_25_fold{fold}_white.partial.npy"
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
        shape=(rows, len(FEATURE_NAMES)))
    offset = 0
    started = time.monotonic()
    for number, (history, online, keep) in enumerate(tasks):
        features = whitening_features(history, online)
        count = len(keep)
        matrix[offset:offset + count] = features[keep]
        offset += count
        if (number + 1) % 250 == 0:
            matrix.flush()
            print(f"Model57 fold {fold} synthetic whitening cache: "
                  f"{number + 1}/{len(tasks)} tasks", flush=True)
    if offset != rows:
        raise ValueError("Model57 synthetic whitening cache is incomplete")
    matrix.flush()
    del matrix
    partial.replace(path)
    write_json(path.with_suffix(".json"), dict(protocol_sha256=fingerprint,
        rows=rows, names=list(FEATURE_NAMES), sha256=sha256(path)))
    print(f"Model57 fold {fold} synthetic whitening complete in "
          f"{time.monotonic()-started:.1f}s", flush=True)
    return cached_white(path, fingerprint, rows)


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"aligned_25_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="aligned_25"))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model57 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != 1000 or len(sy) != 400_000:
        raise ValueError("Changed Model20 synthetic tasks")
    base = read_cache(PARENT, f"aligned_25_fold{fold}_features", dict(
        protocol_sha256=CONFIG["model20_protocol_sha256"], rows=len(sy),
        names=parent["contract"]["names"][fold], tasks=len(tasks)))
    white = synthetic_white(directory, fingerprint, fold, tasks, len(sy))
    sx = np.concatenate((np.asarray(base), np.asarray(white)), axis=1)
    del tasks, base, white
    train_one(directory, protocol, fingerprint, layout, real, plan,
        "aligned_25", (sx, sy, st))
    del sx, sy, st
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
        from models.model57.report57 import report
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        base_real = read_cache(PARENT, "real_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"],
            rows=layout.row_count, names=parent["contract"]["union_names"],
            tasks=int(np.sum(layout.inner_series_fold >= 0))))
        pool = load_real_pool(layout)
        white_real = real_white(directory, fingerprint, layout, pool)
        real = CombinedFeatures(base_real, white_real)
        for fold in CONFIG["pilot_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
