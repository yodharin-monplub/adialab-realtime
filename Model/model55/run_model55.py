"""Fit a synthetic XGBoost voter with causal higher-lag dependence features."""
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
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as voter
from models.model39 import run_model39 as incumbent
from models.model55.dependence import FEATURE_NAMES, LAGS, WINDOWS, STREAMS, dependence_features
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model20_run"]


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
    if (tuple(CONFIG["lags"]) != LAGS or tuple(CONFIG["windows"]) != WINDOWS
            or tuple(CONFIG["streams"]) != STREAMS or len(FEATURE_NAMES) != 30):
        raise ValueError("Dependence feature specification changed")
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
    paths = [HERE / "run_model55.py", HERE / "report55.py", HERE / "dependence.py",
        HERE / "prespec.json", HERE / "readme.txt", PARENT / "protocol.json",
        PARENT / "real_features.cache.json", d26 / "protocol.json",
        d39 / "protocol.json", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model20/report.py", ROOT / "models/model21/run_model21.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "train/X_train.parquet",
        ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    paths += [PARENT / f"aligned_25_fold{fold}_features.cache.json" for fold in range(5)]
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"],
        names=parent["contract"]["names"],
        union_names=parent["contract"]["union_names"],
        extra_names=list(FEATURE_NAMES))
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model55 protocol mismatch")
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
        raise ValueError("Changed Model55 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model55 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model55 input: {relative}")
    return directory, protocol, fingerprint


def cached_extra(path, fingerprint, rows):
    manifest_path = path.with_suffix(".json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    expected = dict(protocol_sha256=fingerprint, rows=rows,
        names=list(FEATURE_NAMES), sha256=sha256(path))
    if manifest != expected:
        raise ValueError(f"Corrupt Model55 feature cache: {path}")
    result = np.load(path, mmap_mode="r", allow_pickle=False)
    if result.shape != (rows, len(FEATURE_NAMES)) or result.dtype != np.float32:
        raise ValueError(f"Wrong Model55 feature-cache shape: {path}")
    return result


def _prefix_parity(history, online):
    full = dependence_features(history, online)
    for cut in (1, 16, 32, 64, 128, 256):
        if cut <= len(online):
            np.testing.assert_array_equal(
                dependence_features(history, online[:cut])[-1], full[cut - 1])
    return full


def real_extra(directory, fingerprint, layout, pool):
    path = directory / "real_extra.npy"
    saved = cached_extra(path, fingerprint, layout.row_count)
    if saved is not None:
        return saved
    partial = directory / "real_extra.partial.npy"
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
        shape=(layout.row_count, len(FEATURE_NAMES)))
    positions = layout.series_position
    total = 0
    started = time.monotonic()
    for number, (position, (history, online)) in enumerate(sorted(pool.items())):
        left, right = np.searchsorted(positions, (position, position + 1))
        if right - left != len(online):
            raise ValueError("Real feature rows do not align with series")
        features = (_prefix_parity(history, online) if number < 3 else
                    dependence_features(history, online))
        matrix[left:right] = features
        total += right - left
        if (number + 1) % 500 == 0:
            matrix.flush()
            print(f"Model55 real dependence cache: {number + 1}/{len(pool)} series", flush=True)
    if total != layout.row_count:
        raise ValueError("Model55 real feature cache is incomplete")
    matrix.flush()
    del matrix
    partial.replace(path)
    write_json(path.with_suffix(".json"), dict(protocol_sha256=fingerprint,
        rows=layout.row_count, names=list(FEATURE_NAMES), sha256=sha256(path)))
    print(f"Model55 real dependence features complete in {time.monotonic()-started:.1f}s", flush=True)
    return cached_extra(path, fingerprint, layout.row_count)


def synthetic_extra(directory, fingerprint, fold, tasks, rows):
    path = directory / f"aligned_25_fold{fold}_extra.npy"
    saved = cached_extra(path, fingerprint, rows)
    if saved is not None:
        return saved
    partial = directory / f"aligned_25_fold{fold}_extra.partial.npy"
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
        shape=(rows, len(FEATURE_NAMES)))
    offset = 0
    started = time.monotonic()
    for number, (history, online, keep) in enumerate(tasks):
        features = dependence_features(history, online)
        count = len(keep)
        matrix[offset:offset + count] = features[keep]
        offset += count
        if (number + 1) % 250 == 0:
            matrix.flush()
            print(f"Model55 fold {fold} synthetic dependence cache: "
                  f"{number + 1}/{len(tasks)} tasks", flush=True)
    if offset != rows:
        raise ValueError("Model55 synthetic feature cache is incomplete")
    matrix.flush()
    del matrix
    partial.replace(path)
    write_json(path.with_suffix(".json"), dict(protocol_sha256=fingerprint,
        rows=rows, names=list(FEATURE_NAMES), sha256=sha256(path)))
    print(f"Model55 fold {fold} synthetic dependence features complete in "
          f"{time.monotonic()-started:.1f}s", flush=True)
    return cached_extra(path, fingerprint, rows)


def fit_one(directory, protocol, fingerprint, layout, pool, base_real, extra_real, fold):
    path = directory / f"aligned_25_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm="aligned_25")
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model55 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    recipe = protocol["contract"]["recipes"][fold]
    names = protocol["contract"]["names"][fold]
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    fit, fitting = fitting_rows(layout, fold, recipe["seed"])
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    if np.any(np.isin(layout.row_fold[fit], (fold, (fold + 1) % 5))):
        raise ValueError("Model55 fitting rows overlap held or stopping fold")
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != 1000 or len(sy) != 400_000:
        raise ValueError("Changed Model20 synthetic task count")
    sx_base = read_cache(PARENT, f"aligned_25_fold{fold}_features", dict(
        protocol_sha256=CONFIG["model20_protocol_sha256"], rows=len(sy),
        names=names, tasks=len(tasks)))
    sx_extra = synthetic_extra(directory, fingerprint, fold, tasks, len(sy))
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    real_x = np.concatenate((np.asarray(base_real[fit][:, columns]),
        np.asarray(extra_real[fit])), axis=1)
    synthetic_x = np.concatenate((np.asarray(sx_base), np.asarray(sx_extra)), axis=1)
    rw, sw, accounting = splice_weights(weights, y, t, sy, st, .25)
    x = np.concatenate((real_x, synthetic_x), axis=0)
    target = np.r_[y, sy]
    sample_weight = np.r_[rw, sw]
    del tasks, real_x, synthetic_x, sx_base, sx_extra, sy, st
    gc.collect()
    params = {key: value for key, value in recipe["hyperparameters"].items() if key != "name"}
    model = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist",
        n_estimators=recipe["rounds"], eval_metric="logloss",
        random_state=recipe["seed"], n_jobs=4, device="cpu", verbosity=0,
        **params)
    started = time.monotonic()
    print(f"Fitting Model55 fold {fold}: {len(target):,} rows, "
          f"{x.shape[1]} features, {recipe['rounds']} rounds", flush=True)
    model.fit(x, target, sample_weight=sample_weight, verbose=False)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        held_x = np.concatenate((np.asarray(base_real[rows][:, columns]),
            np.asarray(extra_real[rows])), axis=1)
        prediction[left:left + len(rows)] = model.predict_proba(held_x)[:, 1]
    model_path = directory / f"aligned_25_fold{fold}.ubj"
    temporary = directory / f"aligned_25_fold{fold}.partial.ubj"
    model.save_model(temporary)
    temporary.replace(model_path)
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    details = directory / f"aligned_25_fold{fold}.fit.json"
    write_json(details, dict(component_tsauc=score, real_rows=len(fit),
        synthetic_rows=400_000, feature_count=x.shape[1], weights=accounting,
        elapsed_seconds=time.monotonic() - started))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta,
        (model_path, details))
    print(f"Finished Model55 fold {fold}: component TSAUC={score:.9f} "
          f"in {time.monotonic()-started:.1f}s", flush=True)
    del model, x, target, sample_weight
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
        from models.model55.report55 import report
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        base_real = read_cache(PARENT, "real_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"],
            rows=layout.row_count, names=protocol["contract"]["union_names"],
            tasks=int(np.sum(layout.inner_series_fold >= 0))))
        pool = load_real_pool(layout)
        extra_real = real_extra(directory, fingerprint, layout, pool)
        for fold in CONFIG["pilot_folds"]:
            fit_one(directory, protocol, fingerprint, layout, pool, base_real, extra_real, fold)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                fit_one(directory, protocol, fingerprint, layout, pool, base_real, extra_real, fold)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
