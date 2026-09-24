"""Execution-corrected wrapper for Model89's frozen Bayesian experiment."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model89 import run_model89 as base

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
ORIGINAL_FIT_ARM = base.fit_arm
ORIGINAL_FINAL_REPORT = base.final_report


def initialize(p62, parent, dependencies, d26, d39, d62, d67, d84, s84):
    if (HERE / "runs/latest.json").exists():
        return base.frozen_study()
    paths = [
        HERE / "run_model90.py", HERE / "prespec.json", HERE / "readme.txt",
        ROOT / "models/model89/run_model89.py", ROOT / "models/model89/bayes_evidence.py",
        ROOT / "models/model89/amplitude_audit.json",
        ROOT / "models/DISCORD_NEXT_MODEL_BRIEF.md",
        base.PARENT20 / "protocol.json", base.PARENT20 / "real_features.cache.json",
        base.PARENT57 / "protocol.json", base.PARENT57 / "real_white.json",
        d62 / "protocol.json", d62 / "summary.json", d67 / "protocol.json",
        d67 / "summary.json", d84 / "protocol.json", d84 / "summary.json",
        d84 / "pilot_decision.json", d26 / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model20/run_model20.py", ROOT / "models/model20/report.py",
        ROOT / "models/model20/features.py", ROOT / "models/model21/run_model21.py",
        ROOT / "models/model57/run_model57.py", ROOT / "models/model57/whitening.py",
        ROOT / "models/model62/run_model62.py", ROOT / "models/model67/run_model67.py",
        ROOT / "models/model84/run_model84.py", ROOT / "models/model10/causal_transforms.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz", ROOT / "tsauc.py",
    ]
    for fold in range(5):
        paths.extend((base.PARENT20 / f"aligned_25_fold{fold}_features.cache.json",
                      base.PARENT57 / f"aligned_25_fold{fold}_white.json",
                      d84 / f"expressive_fold{fold}.json"))
    union = list(p62["contract"]["union_names"]) + list(base.FEATURE_NAMES)
    names = {
        arm: [list(fold_names) + base.arm_extra_names(arm)
              for fold_names in p62["contract"]["names"]]
        for arm in CONFIG["arms"]
    }
    contract = {
        "config": CONFIG, "dependencies": dependencies,
        "hashes": {str(path.relative_to(ROOT)): base.sha256(path) for path in paths},
        "recipes": parent["contract"]["recipes"], "names": names,
        "union_names": union, "model84_protocol_sha256": s84,
        "repair": "project fold-local synthetic columns into global union before selection",
    }
    protocol = {"contract": contract, "plans": p62["plans"],
                "plans_sha256": p62["plans_sha256"]}
    directory = HERE / "runs" / base.digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model90 protocol mismatch")
    if not path.exists():
        base.write_json(path, protocol)
    fingerprint = base.sha256(path)
    base.write_json(HERE / "runs/latest.json", {
        "directory": str(directory.relative_to(ROOT)),
        "protocol_sha256": fingerprint,
    })
    return directory, protocol, fingerprint


def fit_arm(directory, protocol, fingerprint, layout, real, fold, sx, sy, st, arm):
    """Project fold-local synthetic columns into their union positions."""
    names = protocol["contract"]["names"][arm][fold]
    union = protocol["contract"]["union_names"]
    global_columns = np.asarray([union.index(name) for name in names])
    parent_columns = 317
    local_columns = list(range(parent_columns))
    local_columns += [parent_columns + base.FEATURE_NAMES.index(name)
                      for name in base.arm_extra_names(arm)]
    if sx.shape[1] != parent_columns + len(base.FEATURE_NAMES):
        raise ValueError("Changed Model90 fold-local synthetic contract")
    if len(local_columns) != len(global_columns):
        raise ValueError("Model90 local/global projection length mismatch")
    projected = np.empty((sx.shape[0], len(union)), dtype=np.float32)
    projected[:, global_columns] = sx[:, local_columns]
    ORIGINAL_FIT_ARM(directory, protocol, fingerprint, layout, real, fold,
                     projected, sy, st, arm)


def final_report(directory, fingerprint, layout, decision):
    ORIGINAL_FINAL_REPORT(directory, fingerprint, layout, decision)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["model"] = 90
    base.write_json(summary_path, summary)
    results = (directory / "RESULTS.md").read_text().replace("Model89", "Model90")
    (directory / "RESULTS.md").write_text(results)
    (HERE / "RESULTS.md").write_text(results)


def bind():
    base.HERE = HERE
    base.CONFIG = CONFIG
    base.PARENT20 = ROOT / CONFIG["model20_run"]
    base.PARENT57 = ROOT / CONFIG["model57_run"]
    base.initialize = initialize
    base.fit_arm = fit_arm
    base.final_report = final_report


def main():
    bind()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = base.parent_state()
        directory, protocol, fingerprint = initialize(*state)
        layout = base.load_development_layout(ROOT)
        parent = state[1]
        parent_features = base.read_cache(base.PARENT20, "real_features", {
            "protocol_sha256": CONFIG["model20_protocol_sha256"],
            "rows": layout.row_count, "names": parent["contract"]["union_names"],
            "tasks": int(np.sum(layout.inner_series_fold >= 0)),
        })
        white = base.white_source.cached_white(
            base.PARENT57 / "real_white.npy", CONFIG["model57_protocol_sha256"],
            layout.row_count)
        if white is None:
            raise ValueError("Missing Model57 real cache")
        pool = base.load_real_pool(layout)
        evidence = base.real_evidence(directory, fingerprint, layout, pool)
        real = base.CombinedFeatures(parent_features, white, evidence)
        expected = np.asarray(json.loads((state[7] / "summary.json").read_text())[
            "candidate_fold_scores"])
        observed = base.fold_scores(layout, base.processed_signals(
            directory, fingerprint, layout, [])[0])
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)
        print(f"Model90 reconstructed Model84 baseline exactly: {observed.tolist()}", flush=True)
        for fold in CONFIG["pilot_folds"]:
            base.run_fold(directory, protocol, fingerprint, layout, pool, real,
                          fold, CONFIG["arms"])
        decision = base.pilot_decision(directory, fingerprint, layout)
        if not decision["continue_training"]:
            result = ("# Model90: channel-separated tau-integrated Bayes evidence\n\n"
                "No choice met the prespecified +0.0001 pilot gate. Target 0.640 not "
                "reached; confirmation folds remained closed. No cloud activity.\n")
            (directory / "RESULTS.md").write_text(result)
            (HERE / "RESULTS.md").write_text(result)
            print(result, flush=True)
            return
        for fold in CONFIG["confirmation_folds"]:
            base.run_fold(directory, protocol, fingerprint, layout, pool, real,
                          fold, [decision["selected_arm"]])
        final_report(directory, fingerprint, layout, decision)


if __name__ == "__main__":
    main()
