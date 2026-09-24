"""Late-horizon Fisher reselection of the Model20 synthetic XGBoost voter."""
from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.causal_transforms import apply_causal_transform
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.report import ensemble_margin
from models.model20.run_model20 import synthetic_tasks, train_one
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.model53.selection import late_fisher_names
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model20_run"]
BROAD = ROOT / CONFIG["model47_run"]


def parent_state():
    if sha256(PARENT / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 protocol")
    if sha256(BROAD / "protocol.json") != CONFIG["model47_protocol_sha256"]:
        raise ValueError("Changed Model47 protocol")
    d26, _, s26 = voter.frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed incumbent protocol")
    parent = json.loads((PARENT / "protocol.json").read_text())
    broad = json.loads((BROAD / "protocol.json").read_text())
    if digest(parent["plans"]) != parent["plans_sha256"] or broad["plans"] != parent["plans"]:
        raise ValueError("Changed Model20 donor plan or Model47 training rows")
    union = parent["contract"]["union_names"]
    if len(union) != 439 or broad["contract"]["union_names"] != union:
        raise ValueError("Unexpected feature union")
    dependencies = {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")}
    dependencies.update(xgboost=xgb.__version__,
        xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    if dependencies != parent["contract"]["dependencies"]:
        raise ValueError("Use the Model20 dependency environment")
    return parent, union, dependencies, d26, d39


def real_cache(layout, union):
    return read_cache(PARENT, "real_features", dict(
        protocol_sha256=CONFIG["model20_protocol_sha256"],
        rows=layout.row_count, names=union,
        tasks=int(np.sum(layout.inner_series_fold >= 0))))


def initialize(layout, real, parent, union, dependencies, d26, d39):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    names, diagnostics = [], []
    for fold in range(5):
        chosen, result = late_fisher_names(layout, real, parent, fold,
            start=CONFIG["late_start"], stop=CONFIG["late_stop"],
            count=CONFIG["selected_features"])
        names.append(chosen)
        diagnostics.append(result)
        print(f"Fold {fold} late Fisher: {result}", flush=True)
    paths = [HERE / "run_model53.py", HERE / "selection.py", HERE / "prespec.json",
        HERE / "readme.txt", PARENT / "protocol.json", BROAD / "protocol.json",
        PARENT / "real_features.cache.json", d26 / "protocol.json",
        d39 / "protocol.json", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model20/report.py", ROOT / "models/model21/run_model21.py",
        ROOT / "models/model10/sources.py", ROOT / "models/model10/causal_transforms.py",
        ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    paths += [BROAD / f"aligned_25_fold{fold}_features.cache.json" for fold in range(5)]
    contract = dict(config=CONFIG, dependencies=dependencies,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"], names=names, union_names=union,
        selection_diagnostics=diagnostics)
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model53 protocol mismatch")
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
        raise ValueError("Changed Model53 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model53 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model53 input: {relative}")
    return directory, protocol, fingerprint


def candidate_margins(layout, directory, fingerprint, completed_folds):
    m26, m35, m39 = incumbent.margins(layout)
    anti_delta = m35.astype(np.float64) - m26.astype(np.float64)
    d26, _, s26 = voter.frozen_study()
    original = np.empty(layout.row_count, dtype=np.float32)
    candidate = np.empty_like(original)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        original[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        if fold in completed_folds:
            saved = read_checkpoint(directory / f"aligned_25_fold{fold}.npz",
                dict(protocol_sha256=fingerprint, fold=fold, arm="aligned_25"))
            if saved is None:
                raise ValueError(f"Incomplete Model53 fold {fold}")
            validate_predictions(saved, held)
            candidate[held] = saved["prediction"]
        else:
            candidate[held] = original[held]
    values = [m39]
    for weight in CONFIG["blend_weights"]:
        probability = ((1 - weight) * original + weight * candidate).astype(np.float32)
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted = expit(replaced + anti_delta)
        values.append(apply_causal_transform(adjusted, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32))
    return values


def fold_scores(layout, values):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])


def pilot_decision(directory, fingerprint, layout):
    scores = fold_scores(layout, candidate_margins(layout, directory, fingerprint,
        CONFIG["pilot_folds"]))
    pilot = np.asarray(CONFIG["pilot_folds"])
    gains = scores[1:, pilot].mean(axis=1) - scores[0, pilot].mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, pilot_gains=gains.tolist(),
        baseline_scores=scores[0, pilot].tolist(),
        candidate_scores=scores[selected + 1, pilot].tolist(),
        selected_index=selected, selected_weight=CONFIG["blend_weights"][selected],
        continue_training=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model53 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, decision):
    values = candidate_margins(layout, directory, fingerprint, range(5))
    scores = fold_scores(layout, values)
    base, candidate = scores[0], scores[decision["selected_index"] + 1]
    confirm = np.asarray(CONFIG["confirmation_folds"])
    gain = float(candidate[confirm].mean() - base[confirm].mean())
    wins = int(np.sum(candidate[confirm] > base[confirm]))
    scorers = []
    for fold in confirm:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        pair = [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows])
            for value in (values[0], values[decision["selected_index"] + 1])]
        scorers.append((series, pair))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = []
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local = []
        for series, pair in scorers:
            multiplicity = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            weights = np.zeros(len(layout.inner_series_fold))
            weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in pair])
        mean = np.asarray(local).mean(axis=0)
        boot.append(float(mean[1] - mean[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)),
                dict(protocol_sha256=fingerprint, selected_weight=decision["selected_weight"]))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], values[decision["selected_index"] + 1][rows],
            layout.time_online[rows]) - tsauc(layout.target[rows], values[0][rows],
            layout.time_online[rows])))
    confirmation_score, full_score = float(candidate[confirm].mean()), float(candidate.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and
        full_score >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_confirmation_gain"] and
        wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    summary = dict(model=53, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model53: late-horizon Fisher feature reselection\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| model39 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| 256 late-selected features, weight {decision['selected_weight']:.2f} | "
        f"{confirmation_score:.6f} | {full_score:.6f} | {gain:+.6f} | {wins}/3 | "
        f"{lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {decision['pilot_gains']}.\n"
        f"Confirmation split-half gains: {split}.\n"
        "Exploratory held-series OOF; inherited feature union and ensemble are development-informed.\n\n"
        f"{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint, layout):
    decision = pilot_decision(directory, fingerprint, layout)
    if not decision["continue_training"]:
        result = ("# Model53: late-horizon Fisher feature reselection\n\n"
            f"Pilot gains: {dict(zip(CONFIG['blend_weights'], decision['pilot_gains']))}.\n"
            "No weight met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"aligned_25_fold{fold}.npz").exists()
           for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, decision)
    else:
        print(f"Pilot selected Model53 weight {decision['selected_weight']:.2f}; "
              "confirmation fits pending", flush=True)
    return True


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold):
    plan = protocol["plans"][fold]
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"aligned_25_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="aligned_25"))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model53 fold {fold} already complete", flush=True)
        return
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != 1000 or len(sy) != 400_000:
        raise ValueError("Model53 synthetic task count changed")
    union = protocol["contract"]["union_names"]
    broad = read_cache(BROAD, f"aligned_25_fold{fold}_features", dict(
        protocol_sha256=CONFIG["model47_protocol_sha256"],
        rows=len(sy), names=union, tasks=len(tasks)))
    columns = np.asarray([union.index(name) for name in protocol["contract"]["names"][fold]])
    sx = np.asarray(broad[:, columns], dtype=np.float32)
    del tasks, broad
    train_one(directory, protocol, fingerprint, layout, real, plan,
        "aligned_25", (sx, sy, st))
    del sx, sy, st
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "run":
            layout = load_development_layout(ROOT)
            parent, union, dependencies, d26, d39 = parent_state()
            real = real_cache(layout, union)
            directory, protocol, fingerprint = initialize(
                layout, real, parent, union, dependencies, d26, d39)
            pool = load_real_pool(layout)
            for fold in CONFIG["pilot_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
            if report(directory, fingerprint, layout):
                for fold in CONFIG["confirmation_folds"]:
                    run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
                report(directory, fingerprint, layout)
        else:
            directory, _, fingerprint = frozen_study()
            report(directory, fingerprint, load_development_layout(ROOT))


if __name__ == "__main__":
    main()
