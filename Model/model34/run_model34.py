"""TSAUC pairwise additive correction to the frozen Model26 margin."""
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

import lightgbm as lgb
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model11.pipeline import sample_pairs, pairwise_derivatives
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.report import ensemble_margin
from models.model20.run_model20 import fitting_rows
from models.model26 import run_model26 as incumbent
from tsauc import tsauc


HERE = Path(__file__).resolve().parent
PRESPEC = HERE / "prespec.json"
CONFIG = json.loads(PRESPEC.read_text())
MODEL33 = ROOT / "models/model33"
ARMS = ("model26", "pairwise_residual")
FEATURE_COLUMNS = np.asarray([*range(38), 39, 40, 41, *range(42, 55)], dtype=np.int32)


def _model33_state():
    latest = json.loads((MODEL33 / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or fingerprint != CONFIG["model33_protocol_sha256"]:
        raise ValueError("Changed Model33 protocol")
    cache_manifest = json.loads((directory / "level_one.cache.json").read_text())
    cache = directory / "level_one.npy"
    if sha256(cache) != cache_manifest["sha256"]:
        raise ValueError("Changed Model33 level-one cache")
    return directory, fingerprint, cache_manifest


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d33, s33, cache_manifest = _model33_state()
    d26, _, s26 = incumbent.frozen_study()
    files = [
        HERE / "run_model34.py",
        HERE / "report.py",
        HERE / "readme.txt",
        PRESPEC,
        ROOT / "tests/test_model34.py",
        d33 / "protocol.json",
        d33 / "level_one.cache.json",
        d26 / "protocol.json",
        ROOT / "models/model11/pipeline.py",
    ]
    contract = dict(
        config=CONFIG,
        feature_columns=FEATURE_COLUMNS.tolist(),
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        model33_protocol_sha256=s33,
        model33_cache_sha256=cache_manifest["sha256"],
        model26_protocol_sha256=s26,
        dependencies={
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "lightgbm": lgb.__version__,
        },
        python=sys.version,
    )
    protocol = dict(contract=contract)
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model34 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    write_json(HERE / "runs/latest.json", dict(
        directory=str(directory.relative_to(ROOT)), protocol_sha256=sha256(path)
    ))
    return directory, protocol, sha256(path)


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model34 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model34 input: {relative}")
    _model33_state()
    return directory, protocol, fingerprint


def level_one_and_base(layout):
    d33, _, manifest = _model33_state()
    matrix = np.load(d33 / "level_one.npy", mmap_mode="r")
    if matrix.shape != (layout.row_count, 55) or matrix.dtype != np.float32:
        raise ValueError("Invalid Model33 level-one cache dimensions")
    d26, _, s26 = incumbent.frozen_study()
    probability = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        probability[held] = incumbent.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
    margin = ensemble_margin(probability, layout)
    np.testing.assert_array_equal(margin, matrix[:, 38])
    if manifest["contract"]["columns"] != 55:
        raise ValueError("Model33 cache contract changed")
    return matrix, margin


def fit_one(directory, fingerprint, layout, matrix, base_margin, fold):
    path = directory / f"residual_fold{fold}.npz"
    metadata = dict(protocol_sha256=fingerprint, fold=fold, arm="pairwise_residual")
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, metadata)
    if saved is not None:
        validate_predictions(saved, held)
        return
    fit, _ = fitting_rows(layout, fold, CONFIG["seed_base"] + fold)
    if len(fit) != CONFIG["fit_rows"]:
        raise ValueError("Unexpected Model34 fitting row count")
    y = layout.target[fit]
    steps = layout.time_online[fit]
    positive, negative = sample_pairs(y, steps, CONFIG["pairs"], CONFIG["seed_base"] + fold)

    def objective(prediction, dataset):
        return pairwise_derivatives(prediction, positive, negative)

    features = np.asarray(matrix[fit][:, FEATURE_COLUMNS], dtype=np.float32)
    data = lgb.Dataset(
        features,
        label=y,
        init_score=np.asarray(base_margin[fit], dtype=np.float64),
        free_raw_data=False,
    )
    parameters = dict(
        objective=objective,
        metric="None",
        verbosity=-1,
        num_threads=CONFIG["threads"],
        deterministic=True,
        force_col_wise=True,
        seed=CONFIG["seed_base"] + fold,
        num_leaves=CONFIG["num_leaves"],
        learning_rate=CONFIG["learning_rate"],
        min_data_in_leaf=CONFIG["min_data_in_leaf"],
        lambda_l2=CONFIG["reg_lambda"],
        feature_fraction=CONFIG["feature_fraction"],
        max_bin=CONFIG["max_bin"],
    )
    started = time.monotonic()
    print(f"Fitting Model34 fold {fold}: {len(fit):,} rows, {len(positive):,} pairs", flush=True)
    model = lgb.train(parameters, data, num_boost_round=CONFIG["rounds"])
    prediction = np.empty(len(held), dtype=np.float32)
    correction_rms = []
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        correction = model.predict(
            np.asarray(matrix[rows][:, FEATURE_COLUMNS]), raw_score=True, num_threads=CONFIG["threads"]
        )
        prediction[left:left + len(rows)] = expit(base_margin[rows] + correction).astype(np.float32)
        correction_rms.append(float(np.sum(np.square(correction), dtype=np.float64)))
    model_path = directory / f"residual_fold{fold}.txt"
    model.save_model(str(model_path))
    fit_path = directory / f"residual_fold{fold}.fit.json"
    write_json(fit_path, dict(
        score=tsauc(layout.target[held], prediction, layout.time_online[held]),
        real_rows=len(fit), pairs=len(positive),
        correction_rms=float(np.sqrt(sum(correction_rms) / len(held))),
        elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    ))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), metadata, (model_path, fit_path))
    details = json.loads(fit_path.read_text())
    print(
        f"Finished Model34 fold {fold}: TSAUC={details['score']:.9f}, correction RMS={details['correction_rms']:.4f}",
        flush=True,
    )
    del model, data, features, prediction, positive, negative
    gc.collect()


def prediction_checkpoint(directory, fold, fingerprint, held):
    saved = read_checkpoint(
        directory / f"residual_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="pairwise_residual"),
    )
    if saved is None:
        raise RuntimeError(f"Model34 fold {fold} is incomplete")
    validate_predictions(saved, held)
    return saved["prediction"]


def run():
    directory, _, fingerprint = initialize()
    layout = load_development_layout(ROOT)
    matrix, base_margin = level_one_and_base(layout)
    for fold in range(5):
        fit_one(directory, fingerprint, layout, matrix, base_margin, fold)
    from models.model34.report import report
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
            from models.model34.report import report
            report()


if __name__ == "__main__":
    main()
