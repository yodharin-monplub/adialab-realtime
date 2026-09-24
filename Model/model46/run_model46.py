"""Train CatBoost synthetic-data voters and evaluate a frozen ensemble blend."""
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
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.causal_transforms import apply_causal_transform
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.report import ensemble_margin
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model20_run"]


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    if sha256(PARENT / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 protocol")
    d26, _, s26 = voter.frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed upstream protocol")
    parent = json.loads((PARENT / "protocol.json").read_text())
    dependencies = {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "catboost")}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn"):
        if dependencies[name] != parent["contract"]["dependencies"][name]:
            raise ValueError("Use Model20 dependency environment")
    if dependencies["catboost"] != "1.2.10":
        raise ValueError("Use the predeclared CatBoost version")
    paths = [HERE / "run_model46.py", HERE / "prespec.json", HERE / "readme.txt",
             PARENT / "protocol.json", d26 / "protocol.json", d39 / "protocol.json"]
    for fold in range(5):
        paths.append(PARENT / f"aligned_25_fold{fold}_features.cache.json")
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        names=parent["contract"]["names"], union_names=parent["contract"]["union_names"],
        model26_run=str(d26.relative_to(ROOT)))
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model46 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", dict(directory=str(directory.relative_to(ROOT)),
        protocol_sha256=fingerprint))
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model46 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model46 input: {relative}")
    return directory, protocol, fingerprint


def fit_one(directory, protocol, fingerprint, layout, real, plan, synthetic):
    fold = plan["fold"]
    name = "catboost_aligned_25"
    path = directory / f"{name}_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm=name)
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        return
    names = protocol["contract"]["names"][fold]
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    parent_recipe = json.loads((PARENT / "protocol.json").read_text())["contract"]["recipes"][fold]
    fit, fitting = fitting_rows(layout, fold, parent_recipe["seed"])
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    sx, sy, st = synthetic
    rw, sw, accounting = splice_weights(weights, y, t, sy, st, 0.25)
    x = np.concatenate((real[fit][:, columns], sx))
    y = np.r_[y, sy]
    weights = np.r_[rw, sw]
    model = catboost.CatBoostClassifier(loss_function="Logloss", task_type="CPU",
        random_seed=parent_recipe["seed"], thread_count=CONFIG["threads"],
        allow_writing_files=False, verbose=False, **CONFIG["catboost"])
    started = time.monotonic()
    print(f"Fitting Model46 fold {fold}: {len(y):,} rows, {len(columns)} features", flush=True)
    model.fit(x, y, sample_weight=weights)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        prediction[left:left + len(rows)] = model.predict_proba(real[rows][:, columns])[:, 1]
    model_path = directory / f"{name}_fold{fold}.cbm"
    temporary = model_path.with_suffix(".partial.cbm")
    model.save_model(str(temporary))
    temporary.replace(model_path)
    details = directory / f"{name}_fold{fold}.fit.json"
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    write_json(details, dict(component_tsauc=score, real_rows=len(fit),
        synthetic_rows=len(sy), weights=accounting,
        elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta,
        (model_path, details))
    print(f"Finished Model46 fold {fold}: component={score:.9f}", flush=True)
    del model, x
    gc.collect()


def catboost_probability(directory, fold, fingerprint, held):
    name = "catboost_aligned_25"
    saved = read_checkpoint(directory / f"{name}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=name))
    if saved is None:
        raise RuntimeError(f"Incomplete Model46 fold {fold}")
    validate_predictions(saved, held)
    return saved["prediction"]


def candidate_margins(layout, directory, fingerprint):
    model26_margin, model35_margin, model39_margin = incumbent.margins(layout)
    anti_delta = model35_margin.astype(np.float64) - model26_margin.astype(np.float64)
    d26, _, s26 = voter.frozen_study()
    base_probability = np.empty(layout.row_count, dtype=np.float32)
    new_probability = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        base_probability[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        new_probability[held] = catboost_probability(directory, fold, fingerprint, held)
    values = [model39_margin]
    for weight in CONFIG["blend_weights"]:
        probability = ((1 - weight) * base_probability + weight * new_probability).astype(np.float32)
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted_probability = expit(replaced + anti_delta)
        margin = apply_causal_transform(adjusted_probability, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32)
        values.append(margin)
    return values


def report():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    values = candidate_margins(layout, directory, fingerprint)
    pilot = np.asarray(CONFIG["pilot_folds"])
    confirmation = np.asarray(CONFIG["confirmation_folds"])
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])
    pilot_gains = scores[1:, pilot].mean(axis=1) - scores[0, pilot].mean()
    selected_index = int(np.argmax(pilot_gains)) + 1
    selected_weight = CONFIG["blend_weights"][selected_index - 1]
    baseline = scores[0]
    candidate = scores[selected_index]
    gain = float(candidate[confirmation].mean() - baseline[confirmation].mean())
    wins = int(np.sum(candidate[confirmation] > baseline[confirmation]))
    meta = dict(protocol_sha256=fingerprint, selected_weight=selected_weight)
    saved = read_checkpoint(directory / "bootstrap.npz", meta)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in confirmation:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows])
            for value in (values[0], values[selected_index])]))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    completed = len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local = []
        for series, pair in scorers:
            multiplicity = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            if repeat < completed:
                continue
            weights = np.zeros(len(layout.inner_series_fold))
            weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in pair])
        if repeat < completed:
            continue
        mean = np.asarray(local).mean(axis=0)
        boot.append(float(mean[1] - mean[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)), meta)
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    half = (layout.series_ids * 0x9E3779B1) & 1
    confirm_rows = np.isin(layout.row_fold, confirmation)
    split = []
    for value in (0, 1):
        rows = confirm_rows & (half[layout.series_position] == value)
        split.append(float(tsauc(layout.target[rows], values[selected_index][rows],
            layout.time_online[rows]) - tsauc(layout.target[rows], values[0][rows],
            layout.time_online[rows])))
    confirmation_score = float(candidate[confirmation].mean())
    full_score = float(candidate.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and
        full_score >= CONFIG["target_tsauc"] and
        gain >= CONFIG["minimum_confirmation_gain"] and
        wins >= CONFIG["minimum_confirmation_wins"] and
        lower > 0 and min(split) > 0)
    summary = dict(model=46, baseline_fold_scores=baseline.tolist(),
        candidate_fold_scores=candidate.tolist(), pilot_gains=pilot_gains.tolist(),
        selected_weight=selected_weight, confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model46: CatBoost diversity voter", "",
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| model39 | {baseline[confirmation].mean():.6f} | {baseline.mean():.6f} | +0.000000 | 0/3 | baseline | baseline |",
        f"| CatBoost weight {selected_weight:.2f} | {confirmation_score:.6f} | {full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |",
        "", f"Pilot gains by weight: {dict(zip(CONFIG['blend_weights'], pilot_gains.tolist()))}.",
        f"Confirmation split-half gains: {split}.",
        "Local held-series OOF only; prior research reused development folds.",
        "", summary["deployment"], ""]
    result = "\n".join(lines)
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def run():
    directory, protocol, fingerprint = initialize()
    layout = load_development_layout(ROOT)
    parent_contract = json.loads((PARENT / "protocol.json").read_text())["contract"]
    real = read_cache(PARENT, "real_features", dict(
        protocol_sha256=CONFIG["model20_protocol_sha256"],
        rows=layout.row_count, names=parent_contract["union_names"],
        tasks=int(np.sum(layout.inner_series_fold >= 0))))
    pool = load_real_pool(layout)
    for plan in protocol["plans"]:
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        sx = read_cache(PARENT, f"aligned_25_fold{plan['fold']}_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"], rows=len(sy),
            names=parent_contract["names"][plan["fold"]], tasks=len(tasks)))
        fit_one(directory, protocol, fingerprint, layout, real, plan, (sx, sy, st))
        del tasks, sy, st, sx
        gc.collect()
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
            report()


if __name__ == "__main__":
    main()
