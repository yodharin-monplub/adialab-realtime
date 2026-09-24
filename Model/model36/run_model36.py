"""Train the Model20 backbone with appended multi-order AR-Focus features."""
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
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import fitting_rows, reference_weights
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as incumbent
from models.model36.arp_focus_features import N_FEATURES
from models.model36.features import build_cache
from tsauc import tsauc


HERE = Path(__file__).resolve().parent
PRESPEC = HERE / "prespec.json"
CONFIG = json.loads(PRESPEC.read_text())
MODEL20 = ROOT / CONFIG["model20_run"]
ARMS = ("model26", "enhanced_05", "enhanced_10", "enhanced_20")


def _validate_upstream():
    if sha256(MODEL20 / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 protocol")
    d26, p26, s26 = incumbent.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"]:
        raise ValueError("Changed Model26 protocol")
    return d26, p26, s26


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d26, _, s26 = _validate_upstream()
    parent = json.loads((MODEL20 / "protocol.json").read_text())
    files = [HERE / "run_model36.py", HERE / "report.py", HERE / "features.py",
        HERE / "arp_focus_features.py", HERE / "readme.txt", PRESPEC,
        ROOT / "tests/test_model36.py", MODEL20 / "protocol.json",
        MODEL20 / "real_features.cache.json", d26 / "protocol.json"]
    dependencies = {package: importlib.metadata.version(package) for package in
                    ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")}
    dependencies.update(xgboost=xgb.__version__,
                        xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        recipes=parent["contract"]["recipes"], names=parent["contract"]["names"],
        union_names=parent["contract"]["union_names"], new_features=N_FEATURES,
        model20_protocol_sha256=CONFIG["model20_protocol_sha256"],
        model26_protocol_sha256=s26, dependencies=dependencies, python=sys.version)
    protocol = dict(contract=contract)
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model36 protocol mismatch")
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
        raise ValueError("Changed Model36 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model36 input: {relative}")
    _validate_upstream()
    return directory, protocol, fingerprint


def fit_one(directory, protocol, fingerprint, layout, real, new, fold):
    path = directory / f"enhanced_fold{fold}.npz"
    metadata = dict(protocol_sha256=fingerprint, fold=fold, arm="enhanced")
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, metadata)
    if saved is not None:
        validate_predictions(saved, held)
        return
    recipe = protocol["contract"]["recipes"][fold]
    names = protocol["contract"]["names"][fold]
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    fit, fitting = fitting_rows(layout, fold, recipe["seed"])
    if len(fit) != CONFIG["fit_rows"]:
        raise ValueError("Unexpected Model36 fitting row count")
    parent = json.loads((MODEL20 / "protocol.json").read_text())
    np.testing.assert_array_equal(fit, parent["plans"][fold]["fit_rows"])
    y = layout.target[fit]
    weights = reference_weights(y, layout.time_online[fit],
                                layout.target[fitting], layout.time_online[fitting])
    x = np.concatenate((np.asarray(real[fit][:, columns]), np.asarray(new[fit])), axis=1)
    parameters = {key: value for key, value in recipe["hyperparameters"].items() if key != "name"}
    model = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist",
        n_estimators=CONFIG["rounds"], eval_metric="logloss", random_state=recipe["seed"],
        n_jobs=CONFIG["threads"], device="cpu", verbosity=0, **parameters)
    started = time.monotonic()
    print(f"Fitting Model36 fold {fold}: {len(fit):,} rows, {x.shape[1]} features", flush=True)
    model.fit(x, y, sample_weight=weights, verbose=False)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        block = np.concatenate((np.asarray(real[rows][:, columns]), np.asarray(new[rows])), axis=1)
        prediction[left:left + len(rows)] = model.predict_proba(block)[:, 1]
    model_path = directory / f"enhanced_fold{fold}.ubj"
    temporary = directory / f"enhanced_fold{fold}.partial.ubj"
    model.save_model(temporary)
    temporary.replace(model_path)
    fit_path = directory / f"enhanced_fold{fold}.fit.json"
    write_json(fit_path, dict(component_score=tsauc(layout.target[held], prediction,
        layout.time_online[held]), real_rows=len(fit), features=x.shape[1],
        elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), metadata,
                      (model_path, fit_path))
    print(f"Finished Model36 fold {fold}: component={json.loads(fit_path.read_text())['component_score']:.9f}", flush=True)
    del model, x, prediction
    gc.collect()


def raw_enhanced(directory, fold, fingerprint, held):
    saved = read_checkpoint(directory / f"enhanced_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="enhanced"))
    if saved is None:
        raise RuntimeError(f"Model36 fold {fold} incomplete")
    validate_predictions(saved, held)
    return saved["prediction"]


def candidate_probability(base, enhanced, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Invalid Model36 blend weight")
    return ((1.0 - weight) * np.asarray(base, dtype=np.float32)
            + weight * np.asarray(enhanced, dtype=np.float32)).astype(np.float32)


def prediction_checkpoint(directory, arm, fold, fingerprint, held):
    d26, _, s26 = incumbent.frozen_study()
    base = incumbent.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
    if arm == "model26":
        return base
    if arm not in ARMS[1:]:
        raise ValueError("Unknown Model36 arm")
    enhanced = raw_enhanced(directory, fold, fingerprint, held)
    weight = int(arm.rsplit("_", 1)[1]) / 100.0
    return candidate_probability(base, enhanced, weight)


def run():
    directory, protocol, fingerprint = initialize()
    layout = load_development_layout(ROOT)
    pool = load_real_pool(layout)
    new = build_cache(directory, fingerprint, layout, pool, workers=CONFIG["threads"])
    real = read_cache(MODEL20, "real_features", dict(
        protocol_sha256=CONFIG["model20_protocol_sha256"], rows=layout.row_count,
        names=protocol["contract"]["union_names"], tasks=layout.series_count))
    del pool
    for fold in range(5):
        fit_one(directory, protocol, fingerprint, layout, real, new, fold)
    from models.model36.report import report
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
            from models.model36.report import report
            report()


if __name__ == "__main__":
    main()
