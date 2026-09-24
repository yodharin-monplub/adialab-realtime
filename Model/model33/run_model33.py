"""CPU-only nonlinear meta-learner over the frozen primitive OOF bank."""
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

import numpy as np
from scipy.special import logit
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import (
    commit_checkpoint,
    digest,
    read_checkpoint,
    validate_predictions,
)
from models.model20.features import load_real_pool
from models.model20.report import ensemble_margin
from models.model20.run_model20 import fitting_rows, reference_weights
from models.model27.augmentation import CONTEXT_NAMES, contexts
from models.model26 import run_model26 as incumbent
from tsauc import tsauc


HERE = Path(__file__).resolve().parent
PRESPEC = HERE / "prespec.json"
CONFIG = json.loads(PRESPEC.read_text())
MODEL9_MANIFEST = ROOT / "models/model9/runs/fulldata_819a001db6/pairwise_stack/experiment_manifest.json"
MODEL27 = ROOT / "models/model27"
MODEL30 = ROOT / "models/model30"
ARMS = ("model26", "nonlinear_stack")


def safe_logit(probability, epsilon=1e-6):
    bounded = np.clip(np.asarray(probability, dtype=np.float32), epsilon, 1.0 - epsilon)
    result = np.log(bounded) - np.log1p(-bounded)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite probability logit")
    return result.astype(np.float32, copy=False)


def feature_names(primitive_names):
    names = [f"primitive_logit:{name}" for name in primitive_names]
    names += [
        "primitive_logit_mean",
        "primitive_logit_std",
        "primitive_logit_min",
        "primitive_logit_max",
        "model26_final_margin",
        "model27_raw_logit",
        "model30_raw_logit",
        "log1p_time",
    ]
    names += list(CONTEXT_NAMES)
    if len(names) != 55 or len(set(names)) != 55:
        raise ValueError("Model33 feature contract must contain 55 unique columns")
    return names


def _state(model_dir, expected):
    latest = json.loads((model_dir / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or fingerprint != expected:
        raise ValueError(f"Changed {model_dir.name} protocol")
    return directory, fingerprint


def _primitive_contract():
    if sha256(MODEL9_MANIFEST) != CONFIG["model9_primitive_manifest_sha256"]:
        raise ValueError("Changed Model9 primitive manifest")
    manifest = json.loads(MODEL9_MANIFEST.read_text())
    sources = manifest["primitive_sources"]
    if len(sources) != CONFIG["primitive_count"]:
        raise ValueError("Changed primitive source count")
    names = [source["name"] for source in sources]
    feature_names(names)
    return sources


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d27, s27 = _state(MODEL27, CONFIG["model27_protocol_sha256"])
    d30, s30 = _state(MODEL30, CONFIG["model30_protocol_sha256"])
    d26, _, s26 = incumbent.frozen_study()
    sources = _primitive_contract()
    files = [
        HERE / "run_model33.py",
        HERE / "report.py",
        HERE / "readme.txt",
        PRESPEC,
        ROOT / "tests/test_model33.py",
        MODEL9_MANIFEST,
        d27 / "protocol.json",
        d30 / "protocol.json",
        d26 / "protocol.json",
        ROOT / "models/model20/report.py",
        ROOT / "models/model20/run_model20.py",
        ROOT / "models/model27/augmentation.py",
    ]
    dependencies = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy", "scikit-learn")
    }
    dependencies.update(
        xgboost=xgb.__version__,
        xgboost_distribution=importlib.metadata.version("xgboost-cpu"),
    )
    contract = dict(
        config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        primitive_sources=sources,
        feature_names=feature_names([source["name"] for source in sources]),
        model26_protocol_sha256=s26,
        model27_protocol_sha256=s27,
        model30_protocol_sha256=s30,
        dependencies=dependencies,
        python=sys.version,
    )
    protocol = dict(contract=contract)
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model33 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    write_json(
        HERE / "runs/latest.json",
        dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=sha256(path)),
    )
    return directory, protocol, sha256(path)


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model33 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model33 input: {relative}")
    _state(MODEL27, CONFIG["model27_protocol_sha256"])
    _state(MODEL30, CONFIG["model30_protocol_sha256"])
    return directory, protocol, fingerprint


def _raw_checkpoint(directory, fingerprint, stem, arm, fold, held):
    saved = read_checkpoint(
        directory / f"{stem}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=arm),
    )
    if saved is None:
        raise RuntimeError(f"Missing {arm} fold {fold}")
    validate_predictions(saved, held)
    return saved["prediction"]


def _upstream_vectors(layout):
    d26, _, s26 = incumbent.frozen_study()
    d27, s27 = _state(MODEL27, CONFIG["model27_protocol_sha256"])
    d30, s30 = _state(MODEL30, CONFIG["model30_protocol_sha256"])
    base = np.empty(layout.row_count, dtype=np.float32)
    p27 = np.empty_like(base)
    p30 = np.empty_like(base)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        base[held] = incumbent.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        p27[held] = _raw_checkpoint(d27, s27, "matched_context", "matched_context", fold, held)
        p30[held] = _raw_checkpoint(d30, s30, "matched_15", "matched_15", fold, held)
    return ensemble_margin(base, layout), p27, p30


def build_level_one(directory, fingerprint, layout, protocol):
    path = directory / "level_one.npy"
    manifest_path = directory / "level_one.cache.json"
    contract = dict(
        protocol_sha256=fingerprint,
        rows=layout.row_count,
        columns=55,
        feature_names=protocol["contract"]["feature_names"],
    )
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract"] != contract or sha256(path) != manifest["sha256"]:
            raise ValueError("Invalid Model33 level-one cache")
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape != (layout.row_count, 55) or matrix.dtype != np.float32:
            raise ValueError("Wrong Model33 cache shape or dtype")
        return matrix

    temporary = directory / "level_one.partial.npy"
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(layout.row_count, 55)
    )
    sources = protocol["contract"]["primitive_sources"]
    for column, source in enumerate(sources):
        source_path = ROOT / source["path"]
        if sha256(source_path) != source["sha256"]:
            raise ValueError(f"Changed primitive OOF source: {source_path}")
        probability = np.load(source_path, mmap_mode="r", allow_pickle=False)
        if probability.shape != (layout.row_count,):
            raise ValueError(f"Primitive row alignment failed: {source_path}")
        matrix[:, column] = safe_logit(probability)
        print(f"Model33 primitive {column + 1:02d}/{len(sources)}: {source['name']}", flush=True)

    for left in range(0, layout.row_count, 100_000):
        right = min(left + 100_000, layout.row_count)
        block = np.asarray(matrix[left:right, :34], dtype=np.float64)
        matrix[left:right, 34] = block.mean(axis=1)
        matrix[left:right, 35] = block.std(axis=1)
        matrix[left:right, 36] = block.min(axis=1)
        matrix[left:right, 37] = block.max(axis=1)

    base_margin, p27, p30 = _upstream_vectors(layout)
    matrix[:, 38] = base_margin
    matrix[:, 39] = safe_logit(p27)
    matrix[:, 40] = safe_logit(p30)
    matrix[:, 41] = np.log1p(layout.time_online).astype(np.float32)

    pool = load_real_pool(layout)
    history = contexts(pool)
    context_matrix = np.zeros((len(layout.inner_series_fold), len(CONTEXT_NAMES)), dtype=np.float32)
    for position, value in history.items():
        context_matrix[position] = value
    matrix[:, 42:55] = context_matrix[layout.series_position]
    if not np.isfinite(matrix).all():
        raise ValueError("Model33 level-one cache contains non-finite values")
    matrix.flush()
    del matrix, pool, history, context_matrix, base_margin, p27, p30
    gc.collect()
    temporary.replace(path)
    write_json(manifest_path, dict(contract=contract, sha256=sha256(path)))
    return np.load(path, mmap_mode="r")


def fit_one(directory, fingerprint, layout, features, fold):
    path = directory / f"meta_fold{fold}.npz"
    metadata = dict(protocol_sha256=fingerprint, fold=fold, arm="nonlinear_stack")
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, metadata)
    if saved is not None:
        validate_predictions(saved, held)
        return
    fit, fitting = fitting_rows(layout, fold, CONFIG["seed_base"] + fold)
    if len(fit) != CONFIG["fit_rows"]:
        raise ValueError("Unexpected Model33 fitting row count")
    y = layout.target[fit]
    weights = reference_weights(
        y,
        layout.time_online[fit],
        layout.target[fitting],
        layout.time_online[fitting],
    )
    params = dict(
        objective=CONFIG["objective"],
        tree_method=CONFIG["tree_method"],
        max_depth=CONFIG["max_depth"],
        n_estimators=CONFIG["rounds"],
        learning_rate=CONFIG["learning_rate"],
        min_child_weight=CONFIG["min_child_weight"],
        subsample=CONFIG["subsample"],
        colsample_bytree=CONFIG["colsample_bytree"],
        reg_lambda=CONFIG["reg_lambda"],
        reg_alpha=CONFIG["reg_alpha"],
        random_state=CONFIG["seed_base"] + fold,
        n_jobs=CONFIG["threads"],
        device="cpu",
        eval_metric="logloss",
        verbosity=0,
    )
    model = xgb.XGBClassifier(**params)
    started = time.monotonic()
    print(f"Fitting Model33 fold {fold}: {len(fit):,} rows, {features.shape[1]} features", flush=True)
    model.fit(np.asarray(features[fit]), y, sample_weight=weights, verbose=False)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        prediction[left:left + len(rows)] = model.predict_proba(np.asarray(features[rows]))[:, 1]
    model_path = directory / f"meta_fold{fold}.ubj"
    temporary = directory / f"meta_fold{fold}.partial.ubj"
    model.save_model(temporary)
    temporary.replace(model_path)
    fit_path = directory / f"meta_fold{fold}.fit.json"
    write_json(
        fit_path,
        dict(
            score=tsauc(layout.target[held], prediction, layout.time_online[held]),
            real_rows=len(fit),
            features=features.shape[1],
            elapsed_seconds=time.monotonic() - started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )
    commit_checkpoint(
        path,
        dict(prediction=prediction, held_rows=held),
        metadata,
        (model_path, fit_path),
    )
    print(f"Finished Model33 fold {fold}: TSAUC={json.loads(fit_path.read_text())['score']:.9f}", flush=True)
    del model, prediction
    gc.collect()


def prediction_checkpoint(directory, fold, fingerprint, held):
    saved = read_checkpoint(
        directory / f"meta_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="nonlinear_stack"),
    )
    if saved is None:
        raise RuntimeError(f"Model33 fold {fold} is incomplete")
    validate_predictions(saved, held)
    return saved["prediction"]


def run():
    directory, protocol, fingerprint = initialize()
    layout = load_development_layout(ROOT)
    features = build_level_one(directory, fingerprint, layout, protocol)
    for fold in range(5):
        fit_one(directory, fingerprint, layout, features, fold)
    from models.model33.report import report
    report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("initialize", "run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "initialize":
            initialize()
        elif args.command == "run":
            run()
        else:
            from models.model33.report import report
            report()


if __name__ == "__main__":
    main()
