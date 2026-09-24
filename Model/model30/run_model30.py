"""Lower-shift matched augmentation without history-context columns."""
from pathlib import Path
import argparse
import copy
import fcntl
import importlib.metadata
import json
import sys

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import train_one
from models.model21.run_model21 import read_cache, PARENT, PARENT_SHA
from models.model26 import run_model26 as incumbent
from models.model27.run_model27 import synthetic_tasks


HERE = Path(__file__).resolve().parent
MODEL27 = ROOT / "models/model27"
ARMS = ("model26", "matched15_50")
CONFIG = dict(version=1, arms=list(ARMS), fit_arm="matched_15", depth=3, rounds=800,
    synthetic_share=.15, incumbent_weight=.5, new_weight=.5,
    target_tsauc=.64, minimum_gain=.0007, minimum_fold_wins=4,
    bootstrap_replicates=1000, bootstrap_seed=301109, confidence=.95,
    interpretation=("Exploratory matched-augmentation backbone selected after Model27 showed fold-specific "
        "failure from appended history context. Model30 reuses Model27's frozen positive/hard-negative "
        "composites but removes all context columns, lowers synthetic mass from 25% to 15%, uses depth 3, "
        "and fixes a 50/50 blend with Model26. Every prediction remains held-series OOF."))


def model27_state():
    latest = json.loads((MODEL27 / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"]:
        raise ValueError("Changed Model27 protocol")
    return directory, protocol, fingerprint


def candidate_probability(base, new):
    base = np.asarray(base, dtype=np.float32); new = np.asarray(new, dtype=np.float32)
    if base.shape != new.shape or not np.isfinite(base).all() or not np.isfinite(new).all():
        raise ValueError("Invalid Model30 probabilities")
    return np.float32(CONFIG["incumbent_weight"])*base + np.float32(CONFIG["new_weight"])*new


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    model27_dir, model27_protocol, model27_sha = model27_state()
    model26_dir, _, model26_sha = incumbent.frozen_study()
    recipes = copy.deepcopy(model27_protocol["contract"]["recipes"])
    for recipe in recipes:
        recipe["hyperparameters"]["max_depth"] = CONFIG["depth"]
        recipe["rounds"] = CONFIG["rounds"]
    dependencies = {p: importlib.metadata.version(p) for p in
        ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")}
    dependencies.update(xgboost=xgb.__version__,
        xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    files = [HERE / "run_model30.py", HERE / "report.py", HERE / "readme.txt",
        ROOT / "tests/test_model30.py", model27_dir / "protocol.json", model26_dir / "protocol.json"]
    contract = dict(config=CONFIG, recipes=recipes,
        names=model27_protocol["contract"]["names"],
        union_names=model27_protocol["contract"]["union_names"],
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        dependencies=dependencies, python=sys.version,
        model27_protocol_sha256=model27_sha, model26_protocol_sha256=model26_sha)
    protocol = dict(contract=contract)
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if not path.exists(): write_json(path, protocol)
    elif json.loads(path.read_text()) != protocol: raise ValueError("Model30 protocol mismatch")
    write_json(HERE / "runs/latest.json", dict(directory=str(directory.relative_to(ROOT)),
        protocol_sha256=sha256(path)))
    return directory, protocol, sha256(path)


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model30 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model30 input: {relative}")
    return directory, protocol, fingerprint


def prediction_checkpoint(directory, arm, fold, fingerprint, held):
    model26_dir, _, model26_sha = incumbent.frozen_study()
    base = incumbent.prediction_checkpoint(model26_dir, "depth_mix", fold, model26_sha, held)
    if arm == "model26": return base
    if arm != "matched15_50": raise ValueError("Unknown Model30 arm")
    saved = read_checkpoint(directory / f"matched_15_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="matched_15"))
    if saved is None: raise RuntimeError(f"Model30 fold {fold} is incomplete")
    validate_predictions(saved, held)
    return candidate_probability(base, saved["prediction"])


def run():
    directory, protocol, fingerprint = initialize()
    model27_dir, model27_protocol, model27_sha = model27_state()
    layout = load_development_layout(ROOT); pool = load_real_pool(layout)
    real = read_cache(PARENT, "real_features", dict(protocol_sha256=PARENT_SHA,
        rows=layout.row_count, names=protocol["contract"]["union_names"],
        tasks=int(np.sum(layout.inner_series_fold >= 0))))
    from models.model27.augmentation import contexts
    context = contexts(pool)
    for plan in model27_protocol["plans"]:
        fold = plan["fold"]
        meta = dict(protocol_sha256=fingerprint, fold=fold, arm="matched_15")
        if read_checkpoint(directory / f"matched_15_fold{fold}.npz", meta) is not None: continue
        tasks, sy, st, _ = synthetic_tasks(pool, context, plan, layout)
        sx = read_cache(model27_dir, f"matched_fold{fold}_features", dict(
            protocol_sha256=model27_sha, rows=len(sy), names=protocol["contract"]["names"][fold],
            tasks=len(tasks)))
        train_one(directory, protocol, fingerprint, layout, real, plan, "matched_15", (sx, sy, st))
    from models.model30.report import report
    report()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=("initialize","run","report"))
    args=parser.parse_args(); (HERE/"runs").mkdir(parents=True,exist_ok=True)
    with (HERE/"runs/active.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=="initialize": initialize()
        elif args.command=="run": run()
        else:
            from models.model30.report import report
            report()


if __name__ == "__main__": main()
