"""Train Model62's learner with regime-matched postbreak donors."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import digest, read_checkpoint, validate_predictions
from models.model20.features import build_cache, load_real_pool
from models.model20.run_model20 import synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model44.run_model44 import fit_one
from models.model57 import run_model57 as white_source
from models.model62 import run_model62
from models.model67 import run_model67
from models.model79.matching import DESCRIPTOR_NAMES, match_plan

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]
PARENT62 = ROOT / CONFIG["model62_run"]
ARM = "lightgbm_aligned_40"


def parent_state():
    d62, p62, s62 = run_model62.frozen_study()
    d67, _, s67 = run_model67.frozen_study()
    if d62 != PARENT62 or s62 != CONFIG["model62_protocol_sha256"]:
        raise ValueError("Changed Model62 parent")
    if str(d67.relative_to(ROOT)) != CONFIG["model67_run"] or s67 != CONFIG["model67_protocol_sha256"]:
        raise ValueError("Changed Model67 parent")
    for key in ("model20_run", "model20_protocol_sha256", "model57_run",
                "model57_protocol_sha256", "lightgbm", "synthetic_mass"):
        if CONFIG[key] != run_model62.CONFIG[key]:
            raise ValueError(f"Model79 changed fixed Model62 field: {key}")
    parent20 = json.loads((PARENT20 / "protocol.json").read_text())
    return p62, parent20


def make_plans(layout, pool, parent62):
    plans = []
    for fold in range(5):
        original = parent62["plans"][fold]
        donors = {position: values for position, values in pool.items()
                  if layout.inner_series_fold[position] not in (fold, (fold + 1) % 5)}
        pairs, stats = match_plan(donors, layout.tau_index, layout.inner_series_fold,
            original, CONFIG["descriptor_window"], CONFIG["descriptor_scale_floor"],
            CONFIG["max_b_reuse"])
        if len(pairs) != len(original["pairs"]):
            raise ValueError("Model79 changed original pair count")
        if [pair["a"] for pair in pairs] != [pair["a"] for pair in original["pairs"]]:
            raise ValueError("Model79 changed A donor sequence")
        total = sum(pair["length"] for pair in pairs)
        if total != original["synthetic_total"]:
            raise ValueError("Model79 changed synthetic row universe")
        plans.append(dict(fold=fold, fit_rows=original["fit_rows"], pairs=pairs,
            stats=stats, selected=original["selected"], synthetic_total=total))
        print(f"Frozen Model79 fold {fold}: median distance "
              f"{stats['old_distance_quantiles'][0]:.3f} -> "
              f"{stats['matched_distance_quantiles'][0]:.3f}; "
              f"{stats['unique_b']} unique B donors", flush=True)
    return plans


def initialize(parent62, parent20, layout, pool):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model79.py", HERE / "report79.py", HERE / "matching.py",
        HERE / "prespec.json", HERE / "readme.txt", PARENT20 / "protocol.json",
        PARENT20 / "real_features.cache.json", PARENT57 / "protocol.json",
        PARENT57 / "real_white.json", PARENT62 / "protocol.json", PARENT62 / "summary.json",
        ROOT / CONFIG["model67_run"] / "protocol.json",
        ROOT / CONFIG["model67_run"] / "summary.json",
        ROOT / "models/model18/augmentation.py", ROOT / "models/model20/features.py",
        ROOT / "models/model20/run_model20.py", ROOT / "models/model44/run_model44.py",
        ROOT / "models/model57/run_model57.py", ROOT / "models/model57/whitening.py",
        ROOT / "models/model62/report62.py", ROOT / "models/model67/run_model67.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "train/X_train.parquet",
        ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        names=parent62["contract"]["names"], union_names=parent62["contract"]["union_names"],
        base_names=parent20["contract"]["names"], descriptor_names=list(DESCRIPTOR_NAMES),
        dependencies=parent62["contract"]["dependencies"])
    plans = make_plans(layout, pool, parent62)
    protocol = dict(contract=contract, plans=plans, plans_sha256=digest(plans))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model79 protocol mismatch")
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
        raise ValueError("Changed Model79 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model79 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model79 input: {relative}")
    return directory, protocol, fingerprint


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model79 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != 1000 or len(sy) != 400_000:
        raise ValueError("Changed Model79 synthetic task dimensions")
    base = build_cache(directory, f"matched_aligned_fold{fold}_features", tasks, len(sy),
        protocol["contract"]["base_names"][fold], fingerprint, workers=CONFIG["threads"])
    white = white_source.synthetic_white(directory, fingerprint, fold, tasks, len(sy))
    sx = np.concatenate((np.asarray(base), np.asarray(white)), axis=1)
    del tasks, base, white
    fit_one(directory, protocol, fingerprint, layout, real, plan, "aligned_40", (sx, sy, st))
    del sx, sy, st
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent62, parent20 = parent_state()
        layout = load_development_layout(ROOT)
        pool = load_real_pool(layout)
        directory, protocol, fingerprint = initialize(parent62, parent20, layout, pool)
        from models.model79.report79 import report
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        base_real = read_cache(PARENT20, "real_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"], rows=layout.row_count,
            names=parent20["contract"]["union_names"],
            tasks=int(np.sum(layout.inner_series_fold >= 0))))
        white_real = white_source.cached_white(PARENT57 / "real_white.npy",
            CONFIG["model57_protocol_sha256"], layout.row_count)
        if white_real is None:
            raise ValueError("Missing Model57 real whitening cache")
        real = white_source.CombinedFeatures(base_real, white_real)
        for fold in CONFIG["pilot_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
