"""Broaden synthetic splice recipients without changing model or weight share."""
from __future__ import annotations

import argparse
import fcntl
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
from models.model18.augmentation import select_pairs
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import build_cache, load_real_pool
from models.model20.report import ensemble_margin
from models.model20.run_model20 import synthetic_tasks, train_one
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model20_run"]


def initialize(layout, pool):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    if sha256(PARENT / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 protocol")
    d26, _, s26 = voter.frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed incumbent protocol")
    parent = json.loads((PARENT / "protocol.json").read_text())
    files = [HERE / "run_model52.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT / "protocol.json", d26 / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model18/augmentation.py", ROOT / "models/model20/features.py",
        ROOT / "models/model20/run_model20.py", ROOT / "models/model20/report.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        recipes=parent["contract"]["recipes"], names=parent["contract"]["names"],
        union_names=parent["contract"]["union_names"],
        xgboost=xgb.__version__, xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    plans = []
    for fold in range(5):
        donors = {position: series for position, series in pool.items()
            if layout.inner_series_fold[position] not in (fold, (fold + 1) % 5)}
        pairs, stats = select_pairs(donors, layout.tau_index, layout.inner_series_fold,
            fold, CONFIG["pair_seed"] + fold, count=CONFIG["pairs"], reuse=CONFIG["max_b_reuse"])
        if len(pairs) != CONFIG["pairs"]:
            raise ValueError(f"Model52 fold {fold} has too few valid donor pairs: {stats}")
        total = sum(pair["length"] for pair in pairs)
        selected = np.sort(np.random.default_rng(np.random.SeedSequence(
            [CONFIG["pair_seed"] + fold, 1])).choice(
                total, CONFIG["synthetic_rows"], replace=False))
        plans.append(dict(fold=fold, fit_rows=parent["plans"][fold]["fit_rows"],
            pairs=pairs, stats=stats, selected=selected.tolist(), synthetic_total=total))
        print(f"Frozen Model52 fold {fold}: {stats}", flush=True)
    protocol = dict(contract=contract, plans=plans, plans_sha256=digest(plans))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model52 protocol mismatch")
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
        raise ValueError("Changed Model52 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model52 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model52 input: {relative}")
    return directory, protocol, fingerprint


def probabilities(directory, fingerprint, layout, completed_folds):
    d26, _, s26 = voter.frozen_study()
    baseline = np.empty(layout.row_count, dtype=np.float32)
    candidate = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        baseline[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        if fold in completed_folds:
            saved = read_checkpoint(directory / f"aligned_25_fold{fold}.npz",
                dict(protocol_sha256=fingerprint, fold=fold, arm="aligned_25"))
            if saved is None:
                raise ValueError(f"Incomplete Model52 fold {fold}")
            validate_predictions(saved, held)
            candidate[held] = saved["prediction"]
        else:
            candidate[held] = baseline[held]
    return baseline, candidate


def candidate_margins(layout, directory, fingerprint, completed_folds):
    model26_margin, model35_margin, model39_margin = incumbent.margins(layout)
    anti_delta = model35_margin.astype(np.float64) - model26_margin.astype(np.float64)
    baseline, candidate = probabilities(directory, fingerprint, layout, completed_folds)
    values = [model39_margin]
    for weight in CONFIG["blend_weights"]:
        probability = ((1 - weight) * baseline + weight * candidate).astype(np.float32)
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted = expit(replaced + anti_delta)
        margin = apply_causal_transform(adjusted, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32)
        values.append(margin)
    return values


def fold_scores(layout, values):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])


def pilot_decision(directory, fingerprint, layout):
    values = candidate_margins(layout, directory, fingerprint, CONFIG["pilot_folds"])
    scores = fold_scores(layout, values)
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
        raise ValueError("Changed Model52 pilot decision")
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
            layout.time_online[rows]) - tsauc(layout.target[rows], values[0][rows], layout.time_online[rows])))
    confirmation_score = float(candidate[confirm].mean())
    full_score = float(candidate.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and full_score >= CONFIG["target_tsauc"]
        and gain >= CONFIG["minimum_confirmation_gain"] and wins >= CONFIG["minimum_confirmation_wins"]
        and lower > 0 and min(split) > 0)
    summary = dict(model=52, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model52: broader synthetic donor diversity", "",
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| model39 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |",
        f"| 1,800-pair weight {decision['selected_weight']:.2f} | {confirmation_score:.6f} | {full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |",
        "", f"Pilot gains: {decision['pilot_gains']}.", f"Confirmation split-half gains: {split}.",
        "Exploratory held-series OOF; inherited features and ensemble are development-informed.",
        "", summary["deployment"], ""]
    result = "\n".join(lines)
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint):
    layout = load_development_layout(ROOT)
    decision = pilot_decision(directory, fingerprint, layout)
    if not decision["continue_training"]:
        result = ("# Model52: broader synthetic donor diversity\n\n"
            f"Pilot gains: {dict(zip(CONFIG['blend_weights'], decision['pilot_gains']))}.\n"
            "No weight met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"aligned_25_fold{fold}.json").exists() for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, decision)
    else:
        print(f"Pilot selected Model52 weight {decision['selected_weight']:.2f}; confirmation fits pending", flush=True)
    return True


def run(directory, protocol, fingerprint, layout, pool):
    real = np.load(PARENT / "real_features.npy", mmap_mode="r")
    if real.shape != (layout.row_count, len(protocol["contract"]["union_names"])):
        raise ValueError("Model20 real feature cache changed")
    for fold in CONFIG["pilot_folds"]:
        plan = protocol["plans"][fold]
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        sx = build_cache(directory, f"aligned_25_fold{fold}_features", tasks,
            len(sy), protocol["contract"]["names"][fold], fingerprint,
            workers=CONFIG["feature_workers"])
        train_one(directory, protocol, fingerprint, layout, real, plan,
            "aligned_25", (sx, sy, st))
        del tasks, sx, sy, st
    if not report(directory, fingerprint):
        return
    for fold in CONFIG["confirmation_folds"]:
        plan = protocol["plans"][fold]
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        sx = build_cache(directory, f"aligned_25_fold{fold}_features", tasks,
            len(sy), protocol["contract"]["names"][fold], fingerprint,
            workers=CONFIG["feature_workers"])
        train_one(directory, protocol, fingerprint, layout, real, plan,
            "aligned_25", (sx, sy, st))
        del tasks, sx, sy, st
    report(directory, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "run":
            layout = load_development_layout(ROOT)
            pool = load_real_pool(layout)
            directory, protocol, fingerprint = initialize(layout, pool)
            run(directory, protocol, fingerprint, layout, pool)
        else:
            directory, _, fingerprint = frozen_study()
            report(directory, fingerprint)


if __name__ == "__main__":
    main()
