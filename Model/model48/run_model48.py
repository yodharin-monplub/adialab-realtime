"""CPU-only, time-grouped pairwise ranking on the frozen Model20 feature bank."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
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
    d39, _, s39 = incumbent.frozen_study()
    if s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed Model39 protocol")
    files = [HERE / "run_model48.py", HERE / "prespec.json", HERE / "readme.txt",
             PARENT / "protocol.json", d39 / "protocol.json", ROOT / "tsauc.py"]
    contract = dict(config=CONFIG, hashes={str(p.relative_to(ROOT)): sha256(p) for p in files},
        xgboost=xgb.__version__, xgboost_distribution=importlib.metadata.version("xgboost-cpu"))
    protocol = dict(contract=contract)
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model48 protocol mismatch")
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
        raise ValueError("Changed Model48 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model48 input: {relative}")
    return directory, protocol, fingerprint


def parent_protocol():
    return json.loads((PARENT / "protocol.json").read_text())


def synthetic_labels(plan):
    selected = np.asarray(plan["selected"], dtype=np.int64)
    if len(selected) != 400_000 or not np.all(np.diff(selected) > 0):
        raise ValueError("Unexpected Model20 selected synthetic rows")
    y = np.empty(len(selected), dtype=np.uint8)
    t = np.empty(len(selected), dtype=np.int32)
    offset = 0
    for pair in plan["pairs"]:
        left, right = np.searchsorted(selected, [offset, offset + pair["length"]])
        local = selected[left:right] - offset
        y[left:right] = local >= pair["tau_a"]
        t[left:right] = local
        offset += pair["length"]
    if offset != plan["synthetic_total"] or selected[-1] >= offset:
        raise ValueError("Synthetic row accounting differs from Model20")
    return y, t


def fold_arrays(layout, parent, fold):
    plan = parent["plans"][fold]
    names = parent["contract"]["names"][fold]
    union = parent["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    real = np.load(PARENT / "real_features.npy", mmap_mode="r")
    synthetic = np.load(PARENT / f"aligned_25_fold{fold}_features.npy", mmap_mode="r")
    if real.shape != (layout.row_count, len(union)) or synthetic.shape != (400_000, len(names)):
        raise ValueError("Changed Model20 feature-cache dimensions")
    fit = np.asarray(plan["fit_rows"], dtype=np.int64)
    if len(fit) != 400_000 or np.any(np.isin(layout.row_fold[fit], [fold, (fold + 1) % 5])):
        raise ValueError("Fitting rows include held or stopping series")
    sy, st = synthetic_labels(plan)
    selected = np.sort(np.random.default_rng(CONFIG["synthetic_seed"] + fold).choice(
        len(sy), CONFIG["synthetic_rows"], replace=False))
    x = np.concatenate((np.asarray(real[fit][:, columns]), np.asarray(synthetic[selected])))
    y = np.r_[layout.target[fit], sy[selected]]
    step = np.r_[layout.time_online[fit], st[selected]]
    order = np.argsort(step, kind="stable")
    x, y, step = x[order], y[order], step[order]
    _, starts, counts = np.unique(step, return_index=True, return_counts=True)
    positives = np.add.reduceat(y.astype(np.int64), starts)
    group_weights = (positives * (counts - positives)).astype(np.float64)
    group_weights = np.maximum(group_weights, 1.)
    group_weights /= group_weights.mean()
    held = np.flatnonzero(layout.row_fold == fold)
    held_x = np.asarray(real[held][:, columns])
    return x, y, step.astype(np.int32), group_weights.astype(np.float32), held_x, held


def fit_one(directory, fingerprint, layout, parent, fold):
    path = directory / f"rank_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm="time_rank")
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        return
    x, y, qid, weights, held_x, held = fold_arrays(layout, parent, fold)
    recipe = parent["contract"]["recipes"][fold]
    params = {key: value for key, value in recipe["hyperparameters"].items() if key != "name"}
    model = xgb.XGBRanker(objective="rank:pairwise", tree_method="hist",
        n_estimators=CONFIG["rounds"], random_state=recipe["seed"], n_jobs=4,
        device="cpu", verbosity=0, lambdarank_pair_method="mean",
        lambdarank_num_pair_per_sample=CONFIG["pairs_per_sample"], **params)
    started = time.monotonic()
    print(f"Fitting Model48 ranker fold {fold}: {len(y):,} rows, {len(weights)} time groups", flush=True)
    model.fit(x, y, qid=qid, sample_weight=weights, verbose=False)
    train_score = model.predict(x)
    center = float(train_score.mean())
    scale = float(train_score.std())
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("Degenerate ranker score scale")
    prediction = np.clip((model.predict(held_x) - center) / scale, -8, 8).astype(np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite Model48 prediction")
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    model_path = directory / f"rank_fold{fold}.ubj"
    temporary = directory / f"rank_fold{fold}.partial.ubj"
    model.save_model(temporary)
    temporary.replace(model_path)
    details = directory / f"rank_fold{fold}.fit.json"
    write_json(details, dict(component_tsauc=score, center=center, scale=scale,
        rows=len(y), real_rows=400_000, synthetic_rows=CONFIG["synthetic_rows"],
        elapsed_seconds=time.monotonic() - started))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta,
        (model_path, details))
    print(f"Finished Model48 fold {fold}: ranker TSAUC={score:.9f} in {time.monotonic()-started:.1f}s", flush=True)


def rank_scores(directory, fingerprint, layout, folds):
    values = np.full(layout.row_count, np.nan, dtype=np.float32)
    for fold in folds:
        held = np.flatnonzero(layout.row_fold == fold)
        saved = read_checkpoint(directory / f"rank_fold{fold}.npz",
            dict(protocol_sha256=fingerprint, fold=fold, arm="time_rank"))
        if saved is None:
            raise ValueError(f"Missing Model48 fold {fold}")
        validate_predictions(saved, held)
        values[held] = saved["prediction"]
    return values


def scores_for_folds(layout, baseline, rank, folds):
    baseline_scores, candidate_scores = [], []
    for fold in folds:
        rows = layout.row_fold == fold
        baseline_scores.append(tsauc(layout.target[rows], baseline[rows], layout.time_online[rows]))
        candidate_scores.append([tsauc(layout.target[rows], baseline[rows] + weight * rank[rows],
            layout.time_online[rows]) for weight in CONFIG["weights"]])
    return np.asarray(baseline_scores), np.asarray(candidate_scores)


def pilot_decision(directory, fingerprint, layout, baseline):
    rank = rank_scores(directory, fingerprint, layout, CONFIG["pilot_folds"])
    base, candidates = scores_for_folds(layout, baseline, rank, CONFIG["pilot_folds"])
    gains = candidates.mean(axis=0) - base.mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, pilot_baseline_scores=base.tolist(),
        pilot_candidate_scores=candidates.tolist(), pilot_gains=gains.tolist(),
        selected_index=selected, selected_weight=CONFIG["weights"][selected],
        continue_training=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, baseline, decision):
    rank = rank_scores(directory, fingerprint, layout, range(5))
    base, candidates = scores_for_folds(layout, baseline, rank, range(5))
    selected = candidates[:, decision["selected_index"]]
    confirm = np.asarray(CONFIG["confirmation_folds"])
    confirmation = float(selected[confirm].mean())
    full = float(selected.mean())
    gain = float(confirmation - base[confirm].mean())
    wins = int(np.sum(selected[confirm] > base[confirm]))
    boot = []
    scorers = []
    for fold in confirm:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        pair = [WeightedTSAUC(layout.target[rows], values[rows],
            layout.time_online[rows], layout.series_position[rows])
            for values in (baseline, baseline + decision["selected_weight"] * rank)]
        scorers.append((series, pair))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
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
    candidate = baseline + decision["selected_weight"] * rank
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows])
            - tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
    passed = bool(confirmation >= CONFIG["target_tsauc"] and full >= CONFIG["target_tsauc"]
        and gain >= CONFIG["minimum_confirmation_gain"]
        and wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    summary = dict(model=48, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=selected.tolist(), confirmation_score=confirmation,
        full_oof_score=full, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model48: time-grouped pairwise ranker", "",
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| model39 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |",
        f"| ranker weight {decision['selected_weight']:.2f} | {confirmation:.6f} | {full:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |",
        "", f"Pilot gains: {decision['pilot_gains']}.", f"Confirmation ID-half gains: {split}.",
        "Exploratory held-series OOF; inherited Model39 and feature selection are development-informed.",
        "", summary["deployment"], ""]
    result = "\n".join(lines)
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint):
    layout = load_development_layout(ROOT)
    baseline = incumbent.margins(layout)[2]
    decision = pilot_decision(directory, fingerprint, layout, baseline)
    if not decision["continue_training"]:
        result = ("# Model48: time-grouped pairwise ranker\n\n"
            f"Pilot gains by weight: {dict(zip(CONFIG['weights'], decision['pilot_gains']))}.\n"
            "No weight met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"rank_fold{fold}.json").exists() for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, baseline, decision)
    else:
        print(f"Pilot selected ranker weight {decision['selected_weight']:.2f}; confirmation fits pending", flush=True)
    return True


def run(directory, fingerprint):
    layout = load_development_layout(ROOT)
    parent = parent_protocol()
    for fold in CONFIG["pilot_folds"]:
        fit_one(directory, fingerprint, layout, parent, fold)
    if not report(directory, fingerprint):
        return
    for fold in CONFIG["confirmation_folds"]:
        fit_one(directory, fingerprint, layout, parent, fold)
    report(directory, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, _, fingerprint = initialize()
        if args.command == "run":
            run(directory, fingerprint)
        else:
            report(directory, fingerprint)


if __name__ == "__main__":
    main()
