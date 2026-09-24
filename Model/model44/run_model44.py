"""Train a regularized LightGBM voter pair on the frozen synthetic protocol."""
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as incumbent
from models.model39 import run_model39 as exploratory
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model20_run"]
ARMS = tuple(CONFIG["fit_arms"])


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    if sha256(PARENT / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 protocol")
    d26, _, s26 = incumbent.frozen_study()
    d39, _, s39 = exploratory.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed upstream protocol")
    parent = json.loads((PARENT / "protocol.json").read_text())
    dependencies = {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "lightgbm")}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn"):
        if dependencies[name] != parent["contract"]["dependencies"][name]:
            raise ValueError("Use Model20 dependency environment")
    files = [HERE / "run_model44.py", HERE / "report.py", HERE / "readme.txt",
             HERE / "prespec.json", ROOT / "tests/test_model44.py", PARENT / "protocol.json",
             d26 / "protocol.json", d39 / "protocol.json"]
    for fold in range(5):
        files.append(PARENT / f"aligned_25_fold{fold}_features.cache.json")
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        dependencies=dependencies, names=parent["contract"]["names"],
        union_names=parent["contract"]["union_names"], model26_run=str(d26.relative_to(ROOT)))
    protocol = dict(contract=contract, plans=parent["plans"], plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model44 protocol mismatch")
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
        raise ValueError("Changed Model44 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model44 input: {relative}")
    return directory, protocol, fingerprint


def fit_one(directory, protocol, fingerprint, layout, real, plan, arm, synthetic):
    fold = plan["fold"]
    name = f"lightgbm_{arm}"
    path = directory / f"{name}_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm=name)
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        return
    recipe = protocol["contract"]["config"]["lightgbm"]
    names = protocol["contract"]["names"][fold]
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    parent_recipe = json.loads((PARENT / "protocol.json").read_text())["contract"]["recipes"][fold]
    fit, fitting = fitting_rows(layout, fold, parent_recipe["seed"])
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    x = real[fit][:, columns]
    sx, sy, st = synthetic
    rw, sw, accounting = splice_weights(weights, y, t, sy, st, int(arm.split("_")[1]) / 100)
    x, y, weights = np.concatenate((x, sx)), np.r_[y, sy], np.r_[rw, sw]
    params = dict(recipe)
    model = lgb.LGBMClassifier(objective="binary", random_state=parent_recipe["seed"],
        n_jobs=CONFIG["threads"], verbosity=-1, deterministic=True, force_col_wise=True,
        **params)
    started = time.monotonic()
    print(f"Fitting Model44 {arm} fold {fold}: {len(y):,} rows, {len(columns)} features", flush=True)
    model.fit(x, y, sample_weight=weights)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        prediction[left:left + len(rows)] = model.predict_proba(real[rows][:, columns])[:, 1]
    model_path = directory / f"{name}_fold{fold}.txt"
    temporary = model_path.with_suffix(".partial.txt")
    model.booster_.save_model(temporary)
    temporary.replace(model_path)
    details = directory / f"{name}_fold{fold}.fit.json"
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    write_json(details, dict(score=score, real_rows=len(fit), synthetic_rows=len(sy),
        weights=accounting, elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta, (model_path, details))
    print(f"Finished Model44 {arm} fold {fold}: component={score:.9f}", flush=True)
    del model, x
    gc.collect()


def lightgbm_probability(directory, fold, fingerprint, held):
    values = []
    for arm in ARMS:
        name = f"lightgbm_{arm}"
        saved = read_checkpoint(directory / f"{name}_fold{fold}.npz",
            dict(protocol_sha256=fingerprint, fold=fold, arm=name))
        if saved is None:
            raise RuntimeError(f"Incomplete Model44 {arm} fold {fold}")
        validate_predictions(saved, held)
        values.append(saved["prediction"])
    return ((values[0] + values[1]) * np.float32(0.5)).astype(np.float32)


def run():
    directory, protocol, fingerprint = initialize()
    layout = load_development_layout(ROOT)
    parent_contract = json.loads((PARENT / "protocol.json").read_text())["contract"]
    real = read_cache(PARENT, "real_features", dict(protocol_sha256=CONFIG["model20_protocol_sha256"],
        rows=layout.row_count, names=parent_contract["union_names"],
        tasks=int(np.sum(layout.inner_series_fold >= 0))))
    pool = load_real_pool(layout)
    for plan in protocol["plans"]:
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        sx = read_cache(PARENT, f"aligned_25_fold{plan['fold']}_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"], rows=len(sy),
            names=parent_contract["names"][plan["fold"]], tasks=len(tasks)))
        for arm in ARMS:
            fit_one(directory, protocol, fingerprint, layout, real, plan, arm, (sx, sy, st))
        del tasks, sy, st, sx
        gc.collect()
    from models.model44.report import report
    report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "run":
            run()
        else:
            from models.model44.report import report
            report()


if __name__ == "__main__":
    main()
