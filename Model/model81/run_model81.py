"""Train and evaluate a whitened, time-grouped XGBoost pairwise ranker."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model48.run_model48 import synthetic_labels
from models.model57 import run_model57 as white_source
from models.model67 import run_model67
from models.model76.run_model76 import within_time_rank
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]
ARM = "whitened_time_rank"


def validate_rank_predictions(saved, held):
    if (not np.array_equal(saved["held_rows"], held)
            or saved["prediction"].shape != held.shape
            or saved["prediction"].dtype != np.float32
            or not np.isfinite(saved["prediction"]).all()):
        raise ValueError("Invalid Model81 rank scores or held-row alignment")


def parent_state():
    if sha256(PARENT20 / "protocol.json") != CONFIG["model20_protocol_sha256"]:
        raise ValueError("Changed Model20 parent")
    d57, p57, s57 = white_source.frozen_study()
    d67, _, s67 = run_model67.frozen_study()
    if d57 != PARENT57 or s57 != CONFIG["model57_protocol_sha256"]:
        raise ValueError("Changed Model57 parent")
    if str(d67.relative_to(ROOT)) != CONFIG["model67_run"] or s67 != CONFIG["model67_protocol_sha256"]:
        raise ValueError("Changed Model67 parent")
    parent20 = json.loads((PARENT20 / "protocol.json").read_text())
    return parent20, p57


def initialize(parent20, parent57):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model81.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        ROOT / CONFIG["model67_run"] / "protocol.json",
        ROOT / CONFIG["model67_run"] / "summary.json",
        ROOT / "models/model48/run_model48.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model67/run_model67.py", ROOT / "models/model76/run_model76.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    for fold in range(5):
        paths.extend((PARENT20 / f"aligned_25_fold{fold}_features.cache.json",
            PARENT57 / f"aligned_25_fold{fold}_white.json"))
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        names=parent57["contract"]["names"], union_names=parent57["contract"]["union_names"],
        dependencies=dict(xgboost=xgb.__version__,
            xgboost_distribution=importlib.metadata.version("xgboost-cpu")))
    protocol = dict(contract=contract, plans=parent20["plans"],
        plans_sha256=parent20["plans_sha256"], recipes=parent20["contract"]["recipes"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model81 protocol mismatch")
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
        raise ValueError("Changed Model81 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model81 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model81 input: {relative}")
    return directory, protocol, fingerprint


def fold_arrays(layout, protocol, fold):
    plan = protocol["plans"][fold]
    parent20 = json.loads((PARENT20 / "protocol.json").read_text())
    names = parent20["contract"]["names"][fold]
    union = parent20["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    real_base = np.load(PARENT20 / "real_features.npy", mmap_mode="r", allow_pickle=False)
    real_white = np.load(PARENT57 / "real_white.npy", mmap_mode="r", allow_pickle=False)
    synthetic_base = np.load(PARENT20 / f"aligned_25_fold{fold}_features.npy",
        mmap_mode="r", allow_pickle=False)
    synthetic_white = np.load(PARENT57 / f"aligned_25_fold{fold}_white.npy",
        mmap_mode="r", allow_pickle=False)
    fit = np.asarray(plan["fit_rows"], dtype=np.int64)
    if len(fit) != 400_000 or np.any(np.isin(layout.row_fold[fit], [fold, (fold + 1) % 5])):
        raise ValueError("Model81 fitting rows leak held/stopping series")
    selected = np.sort(np.random.default_rng(CONFIG["synthetic_seed"] + fold).choice(
        len(synthetic_base), CONFIG["synthetic_rows"], replace=False))
    sy, st = synthetic_labels(plan)
    real_x = np.concatenate((np.asarray(real_base[fit][:, columns]),
                             np.asarray(real_white[fit])), axis=1)
    synthetic_x = np.concatenate((np.asarray(synthetic_base[selected]),
                                  np.asarray(synthetic_white[selected])), axis=1)
    x = np.concatenate((real_x, synthetic_x))
    y = np.r_[layout.target[fit], sy[selected]].astype(np.int8)
    step = np.r_[layout.time_online[fit], st[selected]].astype(np.int32)
    order = np.argsort(step, kind="stable")
    x, y, step = x[order], y[order], step[order]
    unique = np.unique(step)
    fitting = (layout.row_fold != fold) & (layout.row_fold != (fold + 1) % 5)
    real_y, real_t = layout.target[fitting], layout.time_online[fitting]
    counts = np.bincount(real_t)
    positives = np.bincount(real_t, weights=real_y, minlength=len(counts))
    mass = positives * (counts - positives)
    mass = np.pad(mass, (0, max(0, int(unique.max(initial=0)) + 1 - len(mass))))
    group_weight = np.maximum(mass[unique], 1.0).astype(np.float64)
    group_weight /= group_weight.mean()
    held = np.flatnonzero(layout.row_fold == fold)
    held_x = np.concatenate((np.asarray(real_base[held][:, columns]),
                             np.asarray(real_white[held])), axis=1)
    return x, y, step, group_weight.astype(np.float32), held_x, held


def fit_one(directory, protocol, fingerprint, layout, fold):
    path = directory / f"{ARM}_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm=ARM)
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_rank_predictions(saved, held)
        print(f"Model81 fold {fold} already complete", flush=True)
        return
    x, y, qid, group_weight, held_x, held = fold_arrays(layout, protocol, fold)
    recipe = protocol["recipes"][fold]
    params = {key: value for key, value in recipe["hyperparameters"].items() if key != "name"}
    model = xgb.XGBRanker(objective="rank:pairwise", tree_method="hist",
        n_estimators=CONFIG["rounds"], random_state=recipe["seed"], n_jobs=CONFIG["threads"],
        device="cpu", verbosity=0, lambdarank_pair_method="mean",
        lambdarank_num_pair_per_sample=CONFIG["pairs_per_sample"], **params)
    started = time.monotonic()
    print(f"Fitting Model81 fold {fold}: {len(y):,} rows x {x.shape[1]} features; "
          f"{len(group_weight)} time groups", flush=True)
    model.fit(x, y, qid=qid, sample_weight=group_weight, verbose=False)
    prediction = model.predict(held_x).astype(np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite Model81 ranking margin")
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    model_path = directory / f"{ARM}_fold{fold}.ubj"
    temporary = directory / f"{ARM}_fold{fold}.partial.ubj"
    model.save_model(temporary)
    temporary.replace(model_path)
    details = directory / f"{ARM}_fold{fold}.fit.json"
    write_json(details, dict(component_tsauc=score, rows=len(y), real_rows=400_000,
        synthetic_rows=CONFIG["synthetic_rows"], time_groups=len(group_weight),
        elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta,
        (model_path, details))
    print(f"Finished Model81 fold {fold}: component={score:.9f}, "
          f"elapsed={time.monotonic()-started:.1f}s", flush=True)


def rank_scores(directory, fingerprint, layout, folds):
    result = np.full(layout.row_count, np.nan, dtype=np.float32)
    for fold in folds:
        held = np.flatnonzero(layout.row_fold == fold)
        saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
            dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
        if saved is None:
            raise ValueError(f"Missing Model81 fold {fold}")
        validate_rank_predictions(saved, held)
        result[held] = saved["prediction"]
    return result


def combined(layout, baseline, rank, weight, folds):
    result = np.full(layout.row_count, np.nan, dtype=np.float32)
    for fold in folds:
        rows = layout.row_fold == fold
        base_rank = within_time_rank(baseline[rows], layout.time_online[rows])
        voter_rank = within_time_rank(rank[rows], layout.time_online[rows])
        result[rows] = ((1 - weight) * base_rank + weight * voter_rank).astype(np.float32)
    return result


def fold_scores(layout, value, folds):
    return np.asarray([tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in folds])


def baseline_prediction(layout):
    parents = run_model67.parent_state()
    _, baseline, _ = run_model67.predictions(layout, parents)
    return baseline


def pilot_decision(directory, fingerprint, layout, baseline):
    folds = CONFIG["pilot_folds"]
    rank = rank_scores(directory, fingerprint, layout, folds)
    base = fold_scores(layout, baseline, folds)
    candidates = np.asarray([fold_scores(layout,
        combined(layout, baseline, rank, weight, folds), folds) for weight in CONFIG["weights"]])
    gains = candidates.mean(axis=1) - base.mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, weights=CONFIG["weights"],
        pilot_gains=gains.tolist(), baseline_scores=base.tolist(),
        candidate_scores=candidates[selected].tolist(), selected_index=selected,
        selected_weight=CONFIG["weights"][selected],
        continue_training=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model81 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, baseline, decision):
    folds = range(5)
    rank = rank_scores(directory, fingerprint, layout, folds)
    candidate = combined(layout, baseline, rank, decision["selected_weight"], folds)
    base_scores = fold_scores(layout, baseline, folds)
    candidate_scores = fold_scores(layout, candidate, folds)
    confirm = np.asarray(CONFIG["confirmation_folds"])
    gain = float(candidate_scores[confirm].mean() - base_scores[confirm].mean())
    wins = int(np.sum(candidate_scores[confirm] > base_scores[confirm]))
    scorers = []
    for fold in confirm:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        pair = [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows])
            for value in (baseline, candidate)]
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
                dict(protocol_sha256=fingerprint, selected_index=decision["selected_index"]))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows]) -
                           tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
    confirmation_score = float(candidate_scores[confirm].mean())
    full_score = float(candidate_scores.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and full_score >= CONFIG["target_tsauc"]
        and gain >= CONFIG["minimum_confirmation_gain"] and wins >= CONFIG["minimum_confirmation_wins"]
        and lower > 0 and min(split) > 0)
    summary = dict(model=81, pilot=decision, baseline_fold_scores=base_scores.tolist(),
        candidate_fold_scores=candidate_scores.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model81: whitened metric-weighted pairwise ranker\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base_scores[confirm].mean():.6f} | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| ranker weight {decision['selected_weight']:.2f} | {confirmation_score:.6f} | {full_score:.6f} | "
        f"{gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {dict(zip(CONFIG['weights'], decision['pilot_gains']))}.\n"
        f"Confirmation split-half gains: {split}.\n\n{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint, layout):
    baseline = baseline_prediction(layout)
    decision = pilot_decision(directory, fingerprint, layout, baseline)
    if not decision["continue_training"]:
        summary = dict(model=81, pilot=decision, passes=False,
            deployment="No packaging, upload, or cloud run performed.")
        write_json(directory / "summary.json", summary)
        result = ("# Model81: whitened metric-weighted pairwise ranker\n\n"
            f"Pilot gains: {dict(zip(CONFIG['weights'], decision['pilot_gains']))}.\n"
            "No choice met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"{ARM}_fold{fold}.npz").exists()
           for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, baseline, decision)
    else:
        print(f"Model81 pilot selected ranker weight {decision['selected_weight']:.2f}; "
              "confirmation fits pending", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent20, parent57 = parent_state()
        directory, protocol, fingerprint = initialize(parent20, parent57)
        layout = load_development_layout(ROOT)
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        for fold in CONFIG["pilot_folds"]:
            fit_one(directory, protocol, fingerprint, layout, fold)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                fit_one(directory, protocol, fingerprint, layout, fold)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
