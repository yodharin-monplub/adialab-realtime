"""Fit CatBoost on original-plus-whitened causal features with 40% synthetic mass."""
from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import catboost
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model57 import run_model57 as white_source
from models.model62 import run_model62 as source
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]
PARENT62 = ROOT / CONFIG["model62_run"]
ARM = "catboost_aligned_40"


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
    if CONFIG["synthetic_mass"] != source.CONFIG["synthetic_mass"]:
        raise ValueError("Model68 must use Model62's 40% synthetic mass")
    if any(len(names) != 317 for names in p62["contract"]["names"]):
        raise ValueError("Changed Model62 feature set")
    if importlib.metadata.version("catboost") != "1.2.10":
        raise ValueError("Use frozen CatBoost 1.2.10")
    dependencies = dict(dependencies, catboost="1.2.10")
    return parent, p62, dependencies, d26, d39


def initialize(parent, p62, dependencies, d26, d39):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model68.py", HERE / "report68.py", HERE / "prespec.json",
        HERE / "readme.txt", PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        PARENT62 / "protocol.json", PARENT62 / "summary.json",
        d26 / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model46/run_model46.py", ROOT / "models/model62/run_model62.py",
        ROOT / "models/model62/report62.py", ROOT / "models/model57/run_model57.py",
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
        recipes=parent["contract"]["recipes"], names=p62["contract"]["names"],
        union_names=p62["contract"]["union_names"],
        extra_names=p62["contract"]["extra_names"])
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model68 protocol mismatch")
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
        raise ValueError("Changed Model68 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model68 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model68 input: {relative}")
    return directory, protocol, fingerprint


def fit_one(directory, protocol, fingerprint, layout, real, plan, synthetic):
    fold = plan["fold"]
    path = directory / f"{ARM}_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm=ARM)
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model68 fold {fold} already complete", flush=True)
        return
    names = protocol["contract"]["names"][fold]
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    parent_recipe = protocol["contract"]["recipes"][fold]
    fit, fitting = fitting_rows(layout, fold, parent_recipe["seed"])
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    sx, sy, st = synthetic
    rw, sw, accounting = splice_weights(weights, y, t, sy, st, CONFIG["synthetic_mass"])
    x = np.concatenate((real[fit][:, columns], sx))
    y, weights = np.r_[y, sy], np.r_[rw, sw]
    model = catboost.CatBoostClassifier(loss_function="Logloss", task_type="CPU",
        random_seed=parent_recipe["seed"], thread_count=CONFIG["threads"],
        allow_writing_files=False, verbose=False, **CONFIG["catboost"])
    started = time.monotonic()
    print(f"Fitting Model68 fold {fold}: {len(y):,} rows, {len(columns)} features", flush=True)
    model.fit(x, y, sample_weight=weights)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        prediction[left:left + len(rows)] = model.predict_proba(real[rows][:, columns])[:, 1]
    model_path = directory / f"{ARM}_fold{fold}.cbm"
    temporary = model_path.with_suffix(".partial.cbm")
    model.save_model(str(temporary))
    temporary.replace(model_path)
    details = directory / f"{ARM}_fold{fold}.fit.json"
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    write_json(details, dict(component_tsauc=score, real_rows=len(fit),
        synthetic_rows=len(sy), weights=accounting,
        elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta,
        (model_path, details))
    print(f"Finished Model68 fold {fold}: component={score:.9f}", flush=True)
    del model, x
    gc.collect()


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold, parent):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model68 fold {fold} already complete", flush=True)
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
    sx = np.concatenate((np.asarray(base), np.asarray(white)), axis=1)
    if sx.shape != (len(sy), 317):
        raise ValueError("Model68 synthetic feature alignment failed")
    del tasks, base, white
    fit_one(directory, protocol, fingerprint, layout, real, plan, (sx, sy, st))
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
        from models.model68.report68 import report
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
