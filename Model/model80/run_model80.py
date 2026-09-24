"""Train Model62 after TSAUC-mass-aligned splice-row resampling."""
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
            raise ValueError(f"Model80 changed fixed Model62 field: {key}")
    parent20 = json.loads((PARENT20 / "protocol.json").read_text())
    return p62, parent20


def full_synthetic_cells(plan):
    total = int(plan["synthetic_total"])
    target = np.empty(total, dtype=np.int8)
    time_online = np.empty(total, dtype=np.int32)
    offset = 0
    for pair in plan["pairs"]:
        length = int(pair["length"])
        tau = int(pair["tau_a"])
        local = np.arange(length, dtype=np.int32)
        target[offset:offset + length] = local >= tau
        time_online[offset:offset + length] = local
        offset += length
    if offset != total:
        raise ValueError("Model80 synthetic universe length mismatch")
    return target, time_online


def metric_sample(layout, original, fold):
    target, time_online = full_synthetic_cells(original)
    cell = 2 * time_online.astype(np.int64) + target
    size = int(cell.max(initial=0)) + 1
    capacity = np.bincount(cell, minlength=size)
    fitting = (layout.row_fold != fold) & (layout.row_fold != (fold + 1) % 5)
    real_y = layout.target[fitting]
    real_t = layout.time_online[fitting]
    counts = np.bincount(real_t)
    positives = np.bincount(real_t, weights=real_y, minlength=len(counts))
    pair_mass = positives * (counts - positives)
    desired = np.zeros(size, dtype=np.float64)
    steps = np.arange((size + 1) // 2)
    step_mass = np.pad(pair_mass, (0, max(0, len(steps) - len(pair_mass))))[:len(steps)]
    desired[0::2] = step_mass[:len(desired[0::2])]
    desired[1::2] = step_mass[:len(desired[1::2])]
    per_row = np.divide(desired, capacity, out=np.zeros(size), where=capacity > 0)[cell]
    if np.sum(per_row > 0) < CONFIG["synthetic_rows"]:
        raise ValueError("Too few positive-mass synthetic rows")
    rng = np.random.default_rng(np.random.SeedSequence([CONFIG["sampling_seed"] + fold, 1]))
    selected = np.sort(rng.choice(len(cell), CONFIG["synthetic_rows"], replace=False,
                                  p=per_row / per_row.sum()))
    old_selected = np.asarray(original["selected"], dtype=np.int64)
    useful = (capacity > 0) & (desired > 0)
    old_count = np.bincount(cell[old_selected], minlength=size)
    new_count = np.bincount(cell[selected], minlength=size)
    old_ratio = np.divide(old_count, desired, out=np.zeros(size), where=desired > 0)
    new_ratio = np.divide(new_count, desired, out=np.zeros(size), where=desired > 0)
    stats = dict(rows=len(selected), possible_rows=len(cell),
        old_count_mass_correlation=float(np.corrcoef(old_count[useful], desired[useful])[0, 1]),
        new_count_mass_correlation=float(np.corrcoef(new_count[useful], desired[useful])[0, 1]),
        old_count_per_mass_cv=float(np.std(old_ratio[useful]) / np.mean(old_ratio[useful])),
        new_count_per_mass_cv=float(np.std(new_ratio[useful]) / np.mean(new_ratio[useful])),
        old_nonempty_cells=int(np.sum(old_count > 0)), new_nonempty_cells=int(np.sum(new_count > 0)))
    return selected.tolist(), stats


def make_plans(layout, parent62):
    plans = []
    for fold, original in enumerate(parent62["plans"]):
        selected, stats = metric_sample(layout, original, fold)
        plans.append(dict(fold=fold, fit_rows=original["fit_rows"], pairs=original["pairs"],
            stats=stats, selected=selected, synthetic_total=original["synthetic_total"]))
        print(f"Frozen Model80 fold {fold}: count/mass correlation "
              f"{stats['old_count_mass_correlation']:.3f} -> "
              f"{stats['new_count_mass_correlation']:.3f}; ratio CV "
              f"{stats['old_count_per_mass_cv']:.2f} -> "
              f"{stats['new_count_per_mass_cv']:.2f}", flush=True)
    return plans


def initialize(parent62, parent20, layout):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model80.py", HERE / "report80.py", HERE / "prespec.json",
        HERE / "readme.txt", PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        PARENT62 / "protocol.json", PARENT62 / "summary.json",
        ROOT / CONFIG["model67_run"] / "protocol.json",
        ROOT / CONFIG["model67_run"] / "summary.json",
        ROOT / "models/model20/features.py", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model44/run_model44.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model57/whitening.py", ROOT / "models/model62/report62.py",
        ROOT / "models/model67/run_model67.py", ROOT / "models/model79/report79.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        names=parent62["contract"]["names"], union_names=parent62["contract"]["union_names"],
        base_names=parent20["contract"]["names"], dependencies=parent62["contract"]["dependencies"])
    plans = make_plans(layout, parent62)
    protocol = dict(contract=contract, plans=plans, plans_sha256=digest(plans))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model80 protocol mismatch")
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
        raise ValueError("Changed Model80 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model80 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model80 input: {relative}")
    return directory, protocol, fingerprint


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model80 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != 1000 or len(sy) != CONFIG["synthetic_rows"]:
        raise ValueError("Changed Model80 synthetic task dimensions")
    base = build_cache(directory, f"metric_aligned_fold{fold}_features", tasks, len(sy),
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
        directory, protocol, fingerprint = initialize(parent62, parent20, layout)
        from models.model80.report80 import report
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        pool = load_real_pool(layout)
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
