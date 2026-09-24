"""Train depth-5 voters and combine depths 3, 4, and 5."""
from __future__ import annotations
import argparse
import copy
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as incumbent
from models.model38 import run_model38 as smoother
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model20_run"]
ARMS = tuple(CONFIG["fit_arms"])


def initialize():
    if (HERE / "runs/latest.json").exists(): return frozen_study()
    if sha256(PARENT / "protocol.json") != CONFIG["model20_protocol_sha256"]: raise ValueError("Changed Model20 protocol")
    d26, p26, s26 = incumbent.frozen_study(); d38, _, s38 = smoother.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s38 != CONFIG["model38_protocol_sha256"]:
        raise ValueError("Changed upstream protocol")
    parent = json.loads((PARENT / "protocol.json").read_text())
    dependencies = {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")}
    dependencies.update(xgboost=xgb.__version__, xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    if dependencies != parent["contract"]["dependencies"]: raise ValueError("Use Model20 dependency environment")
    recipes = copy.deepcopy(p26["contract"]["recipes"])
    for recipe in recipes: recipe["hyperparameters"]["max_depth"] = CONFIG["new_depth"]
    files = [HERE / "run_model43.py", HERE / "report.py", HERE / "readme.txt", HERE / "prespec.json",
             ROOT / "tests/test_model43.py", PARENT / "protocol.json", d26 / "protocol.json", d38 / "protocol.json"]
    for fold in range(5):
        files += [d26 / f"aligned_25_fold{fold}.npz", d26 / f"aligned_60_fold{fold}.npz",
                  PARENT / f"aligned_25_fold{fold}_features.cache.json"]
    contract = dict(config=CONFIG, hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        dependencies=dependencies, recipes=recipes, names=parent["contract"]["names"],
        union_names=parent["contract"]["union_names"], model26_run=str(d26.relative_to(ROOT)))
    protocol = dict(contract=contract, plans=parent["plans"], plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]; directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError("Model43 protocol mismatch")
    if not path.exists(): write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=fingerprint))
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text()); directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text()); fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model43 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected: raise ValueError(f"Changed frozen Model43 input: {relative}")
    return directory, protocol, fingerprint


def fit_one(directory, protocol, fingerprint, layout, real, plan, arm, synthetic):
    fold = plan["fold"]; name = f"depth5_{arm}"; path = directory / f"{name}_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm=name); held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None: validate_predictions(saved, held); return
    recipe = protocol["contract"]["recipes"][fold]; names = protocol["contract"]["names"][fold]
    union = protocol["contract"]["union_names"]; columns = np.asarray([union.index(n) for n in names])
    fit, fitting = fitting_rows(layout, fold, recipe["seed"]); np.testing.assert_array_equal(fit, plan["fit_rows"])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    x = real[fit][:, columns]; sx, sy, st = synthetic
    rw, sw, accounting = splice_weights(weights, y, t, sy, st, int(arm.split("_")[1]) / 100)
    x, y, weights = np.concatenate((x, sx)), np.r_[y, sy], np.r_[rw, sw]
    params = {key: value for key, value in recipe["hyperparameters"].items() if key != "name"}
    model = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", n_estimators=recipe["rounds"],
        eval_metric="logloss", random_state=recipe["seed"], n_jobs=CONFIG["threads"], device="cpu", verbosity=0, **params)
    started = time.monotonic()
    print(f"Fitting Model43 {arm} fold {fold}: {len(y):,} rows, {len(columns)} features, depth={CONFIG['new_depth']}", flush=True)
    model.fit(x, y, sample_weight=weights, verbose=False)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left+50_000]; prediction[left:left+len(rows)] = model.predict_proba(real[rows][:, columns])[:, 1]
    model_path = directory / f"{name}_fold{fold}.ubj"; temporary = model_path.with_suffix(".partial.ubj")
    model.save_model(temporary); temporary.replace(model_path)
    details = directory / f"{name}_fold{fold}.fit.json"
    write_json(details, dict(score=tsauc(layout.target[held], prediction, layout.time_online[held]),
        real_rows=len(fit), synthetic_rows=len(sy), weights=accounting, elapsed_seconds=time.monotonic()-started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta, (model_path, details))
    print(f"Finished Model43 {arm} fold {fold}: component={tsauc(layout.target[held], prediction, layout.time_online[held]):.9f}", flush=True)
    del model, x; gc.collect()


def depth5_pair(directory, fold, fingerprint, held):
    local = []
    for arm in ARMS:
        name = f"depth5_{arm}"
        saved = read_checkpoint(directory / f"{name}_fold{fold}.npz", dict(protocol_sha256=fingerprint, fold=fold, arm=name))
        if saved is None: raise RuntimeError(f"Incomplete Model43 {arm} fold {fold}")
        validate_predictions(saved, held); local.append(saved["prediction"])
    return ((local[0] + local[1]) * np.float32(.5)).astype(np.float32)


def equal_depth_probability(directory, fold, fingerprint, held):
    d26, _, s26 = incumbent.frozen_study()
    depth3 = incumbent.prediction_checkpoint(None, "model23", fold, None, held)
    depth4_values = []
    for arm in ARMS:
        saved = read_checkpoint(d26 / f"{arm}_fold{fold}.npz",
            dict(protocol_sha256=s26, fold=fold, arm=arm))
        if saved is None: raise RuntimeError(f"Incomplete Model26 {arm} fold {fold}")
        validate_predictions(saved, held); depth4_values.append(saved["prediction"])
    d4a, d4b = depth4_values
    depth4 = ((d4a + d4b) * np.float32(.5)).astype(np.float32)
    depth5 = depth5_pair(directory, fold, fingerprint, held)
    weights = np.asarray(CONFIG["depth_weights"], dtype=np.float32)
    if not np.isclose(weights.sum(), 1): raise ValueError("Depth weights do not sum to one")
    return (weights[0]*depth3 + weights[1]*depth4 + weights[2]*depth5).astype(np.float32)


def run():
    directory, protocol, fingerprint = initialize(); layout = load_development_layout(ROOT)
    parent_contract = json.loads((PARENT / "protocol.json").read_text())["contract"]
    real = read_cache(PARENT, "real_features", dict(protocol_sha256=CONFIG["model20_protocol_sha256"],
        rows=layout.row_count, names=parent_contract["union_names"], tasks=int(np.sum(layout.inner_series_fold >= 0))))
    pool = load_real_pool(layout)
    for plan in protocol["plans"]:
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        sx = read_cache(PARENT, f"aligned_25_fold{plan['fold']}_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"], rows=len(sy),
            names=parent_contract["names"][plan["fold"]], tasks=len(tasks)))
        for arm in ARMS: fit_one(directory, protocol, fingerprint, layout, real, plan, arm, (sx, sy, st))
        del tasks, sy, st, sx; gc.collect()
    from models.model43.report import report
    report()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args(); (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "run": run()
        else:
            from models.model43.report import report
            report()
if __name__ == "__main__": main()
