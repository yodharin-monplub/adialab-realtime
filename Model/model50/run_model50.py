"""Train a recent-break onset classifier and retain its evidence causally."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
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
from models.model20.run_model20 import reference_weights, splice_weights
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
    files = [HERE / "run_model50.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model20/run_model20.py", ROOT / "tsauc.py"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        xgboost=xgb.__version__, xgboost_distribution=importlib.metadata.version("xgboost-cpu")))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model50 protocol mismatch")
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
        raise ValueError("Changed Model50 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model50 input: {relative}")
    return directory, protocol, fingerprint


def recent_labels(layout, rows):
    rows = np.asarray(rows, dtype=np.int64)
    tau = layout.tau_index[layout.series_position[rows]]
    step = layout.time_online[rows]
    return ((tau >= 0) & (step >= tau) & (step - tau < CONFIG["recent_window"])).astype(np.uint8)


def synthetic_labels(plan):
    selected = np.asarray(plan["selected"], dtype=np.int64)
    y = np.empty(len(selected), dtype=np.uint8)
    t = np.empty(len(selected), dtype=np.int32)
    offset = 0
    for pair in plan["pairs"]:
        left, right = np.searchsorted(selected, [offset, offset + pair["length"]])
        local = selected[left:right] - offset
        tau = pair["tau_a"]
        y[left:right] = (local >= tau) & (local < tau + CONFIG["recent_window"])
        t[left:right] = local
        offset += pair["length"]
    if len(selected) != 400_000 or offset != plan["synthetic_total"] or selected[-1] >= offset:
        raise ValueError("Synthetic row accounting differs from Model20")
    return y, t


def fitting_rows(layout, fold):
    fitting = (layout.row_fold != fold) & (layout.row_fold != (fold + 1) % 5)
    eligible = np.flatnonzero(fitting)
    recent = recent_labels(layout, eligible).astype(bool)
    positive = eligible[recent]
    negative = np.random.default_rng(CONFIG["negative_seed"] + fold).choice(
        eligible[~recent], len(positive), replace=False)
    rows = np.sort(np.r_[positive, negative])
    if not fitting[rows].all() or len(rows) != 2 * len(positive):
        raise ValueError("Recent-onset fitting rows violate fold contract")
    return rows, fitting, len(positive)


def fit_one(directory, fingerprint, layout, parent, fold):
    path = directory / f"onset_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm="recent_onset")
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        return
    plan = parent["plans"][fold]
    recipe = parent["contract"]["recipes"][fold]
    names = parent["contract"]["names"][fold]
    union = parent["contract"]["union_names"]
    cols = np.asarray([union.index(name) for name in names])
    real = np.load(PARENT / "real_features.npy", mmap_mode="r")
    synthetic = np.load(PARENT / f"aligned_25_fold{fold}_features.npy", mmap_mode="r")
    if real.shape != (layout.row_count, len(union)) or synthetic.shape != (400_000, len(names)):
        raise ValueError("Model20 cached feature dimensions changed")
    fit, fitting, positive_count = fitting_rows(layout, fold)
    y, t = recent_labels(layout, fit), layout.time_online[fit]
    reference_rows = np.flatnonzero(fitting)
    reference_y = recent_labels(layout, reference_rows)
    original = reference_weights(y, t, reference_y, layout.time_online[reference_rows])
    sy, st = synthetic_labels(plan)
    rw, sw, accounting = splice_weights(original, y, t, sy, st,
        CONFIG["synthetic_mass_share"])
    x = np.concatenate((np.asarray(real[fit][:, cols]), np.asarray(synthetic)), axis=0)
    labels = np.r_[y, sy]
    weights = np.r_[rw, sw]
    params = {key: value for key, value in recipe["hyperparameters"].items() if key != "name"}
    model = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist",
        n_estimators=recipe["rounds"], eval_metric="logloss", random_state=recipe["seed"],
        n_jobs=4, device="cpu", verbosity=0, **params)
    started = time.monotonic()
    print(f"Fitting Model50 fold {fold}: {len(fit):,} real ({positive_count:,} recent) + {len(sy):,} synthetic", flush=True)
    model.fit(x, labels, sample_weight=weights, verbose=False)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        prediction[left:left + 50_000] = model.predict_proba(
            np.asarray(real[held[left:left + 50_000]][:, cols]))[:, 1]
    validate_predictions(dict(prediction=prediction, held_rows=held), held)
    onset_score = tsauc(recent_labels(layout, held), prediction, layout.time_online[held])
    model_path = directory / f"onset_fold{fold}.ubj"
    temporary = directory / f"onset_fold{fold}.partial.ubj"
    model.save_model(temporary)
    temporary.replace(model_path)
    details = directory / f"onset_fold{fold}.fit.json"
    write_json(details, dict(recent_onset_tsauc=onset_score, real_rows=len(fit),
        real_recent_rows=positive_count, synthetic_rows=len(sy),
        synthetic_recent_rows=int(sy.sum()), weights=accounting,
        elapsed_seconds=time.monotonic() - started))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta,
        (model_path, details))
    print(f"Finished Model50 fold {fold}: recent-onset TSAUC={onset_score:.9f} in {time.monotonic()-started:.1f}s", flush=True)


def probabilities(directory, fingerprint, layout, completed_folds):
    probability = np.full(layout.row_count, .5, dtype=np.float32)
    for fold in completed_folds:
        held = np.flatnonzero(layout.row_fold == fold)
        saved = read_checkpoint(directory / f"onset_fold{fold}.npz",
            dict(protocol_sha256=fingerprint, fold=fold, arm="recent_onset"))
        if saved is None:
            raise ValueError(f"Incomplete Model50 fold {fold}")
        validate_predictions(saved, held)
        probability[held] = saved["prediction"]
    return probability


def causal_peak(probability, series_position):
    p = np.asarray(probability, dtype=np.float64)
    owner = np.asarray(series_position)
    if p.shape != owner.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid onset probabilities")
    raw = np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
    raw = np.clip(raw, -6, 6)
    signal = np.empty(len(p), dtype=np.float32)
    decay = math.exp(math.log(.5) / CONFIG["decay_half_life"])
    state = 0.0
    previous = -1
    for index in range(len(p)):
        if owner[index] != previous:
            state = 0.0
            previous = owner[index]
        state = max(0.0, raw[index], decay * state)
        signal[index] = state
    return signal


def score_matrix(layout, baseline, signal):
    values = [baseline]
    values += [(baseline.astype(np.float64) + weight * signal).astype(np.float32)
        for weight in CONFIG["blend_weights"]]
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])
    return values, scores


def pilot_decision(directory, fingerprint, layout, baseline):
    probability = probabilities(directory, fingerprint, layout, CONFIG["pilot_folds"])
    signal = causal_peak(probability, layout.series_position)
    _, scores = score_matrix(layout, baseline, signal)
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
        raise ValueError("Changed Model50 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, baseline, decision):
    probability = probabilities(directory, fingerprint, layout, range(5))
    signal = causal_peak(probability, layout.series_position)
    values, scores = score_matrix(layout, baseline, signal)
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
    summary = dict(model=50, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model50: recent-onset specialist with causal peak memory", "",
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| model39 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |",
        f"| onset weight {decision['selected_weight']:.2f} | {confirmation_score:.6f} | {full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |",
        "", f"Pilot gains: {decision['pilot_gains']}.", f"Confirmation split-half gains: {split}.",
        "Exploratory held-series OOF; inherited features and Model39 are development-informed.",
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
        result = ("# Model50: recent-onset specialist with causal peak memory\n\n"
            f"Pilot gains: {dict(zip(CONFIG['blend_weights'], decision['pilot_gains']))}.\n"
            "No weight met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"onset_fold{fold}.json").exists() for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, baseline, decision)
    else:
        print(f"Pilot selected Model50 weight {decision['selected_weight']:.2f}; confirmation fits pending", flush=True)
    return True


def run(directory, fingerprint):
    layout = load_development_layout(ROOT)
    parent = json.loads((PARENT / "protocol.json").read_text())
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
