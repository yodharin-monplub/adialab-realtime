"""Test lower- and higher-capacity LightGBM voters around Model62."""
from __future__ import annotations

import fcntl
import gc
import json
from pathlib import Path
import resource
import sys
import time

import lightgbm as lgb
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
from models.model57 import run_model57 as white_source
from models.model62 import run_model62 as source
from models.model67 import run_model67 as baseline_source
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]


def parent_state():
    d62, p62, s62 = source.frozen_study()
    d67, _, s67 = baseline_source.frozen_study()
    if d62 != ROOT / CONFIG["model62_run"] or s62 != CONFIG["model62_protocol_sha256"]:
        raise ValueError("Changed Model62 parent")
    if d67 != ROOT / CONFIG["model67_run"] or s67 != CONFIG["model67_protocol_sha256"]:
        raise ValueError("Changed Model67 parent")
    parent, _, dependencies, d26, d39 = source.parent_state()
    if (sha256(PARENT20 / "protocol.json") != CONFIG["model20_protocol_sha256"] or
            sha256(PARENT57 / "protocol.json") != CONFIG["model57_protocol_sha256"]):
        raise ValueError("Changed feature-cache parent")
    if any(len(names) != 317 for names in p62["contract"]["names"]):
        raise ValueError("Changed Model62 feature count")
    return p62, parent, dependencies, d26, d39, d62, d67


def initialize(p62, parent, dependencies, d26, d39, d62, d67):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model84.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        d62 / "protocol.json", d62 / "summary.json", d67 / "protocol.json", d67 / "summary.json",
        d26 / "protocol.json", d39 / "protocol.json", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model20/report.py", ROOT / "models/model20/features.py",
        ROOT / "models/model21/run_model21.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model62/run_model62.py", ROOT / "models/model67/run_model67.py",
        ROOT / "models/model10/causal_transforms.py", ROOT / "models/model10/sources.py",
        ROOT / "models/report_next_models.py", ROOT / "train/X_train.parquet",
        ROOT / "train/y_train_index.parquet", ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "tsauc.py"]
    for fold in range(5):
        paths.extend((PARENT20 / f"aligned_25_fold{fold}_features.cache.json",
            PARENT57 / f"aligned_25_fold{fold}_white.json"))
    protocol = {"contract": {"config": CONFIG, "dependencies": dependencies,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "recipes": parent["contract"]["recipes"], "names": p62["contract"]["names"],
        "union_names": p62["contract"]["union_names"]}, "plans": p62["plans"],
        "plans_sha256": p62["plans_sha256"]}
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model84 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", {"directory": str(directory.relative_to(ROOT)),
        "protocol_sha256": fingerprint})
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model84 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model84 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model84 input: {relative}")
    return directory, protocol, fingerprint


def fit_arm(directory, protocol, fingerprint, layout, real, fold, sx, sy, st, arm):
    held = np.flatnonzero(layout.row_fold == fold)
    path = directory / f"{arm}_fold{fold}.npz"
    meta = {"protocol_sha256": fingerprint, "fold": fold, "arm": arm}
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model84 {arm} fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    rw, sw, accounting = splice_weights(weights, y, t, sy, st, CONFIG["synthetic_mass"])
    x = np.concatenate((real[fit][:, columns], sx))
    labels, sample_weights = np.r_[y, sy], np.r_[rw, sw]
    model = lgb.LGBMClassifier(objective="binary", random_state=seed,
        n_jobs=CONFIG["threads"], verbosity=-1, deterministic=True, force_col_wise=True,
        **CONFIG["arms"][arm])
    started = time.monotonic()
    print(f"Fitting Model84 {arm} fold {fold}: {len(labels):,} rows", flush=True)
    model.fit(x, labels, sample_weight=sample_weights)
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        prediction[left:left + len(rows)] = model.predict_proba(real[rows][:, columns])[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("Invalid Model84 prediction")
    model_path = directory / f"{arm}_fold{fold}.txt"
    temporary = model_path.with_suffix(".partial.txt")
    model.booster_.save_model(temporary)
    temporary.replace(model_path)
    details = directory / f"{arm}_fold{fold}.fit.json"
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    write_json(details, {"score": score, "real_rows": len(fit), "synthetic_rows": len(sy),
        "weights": accounting, "elapsed_seconds": time.monotonic() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    commit_checkpoint(path, {"prediction": prediction, "held_rows": held}, meta,
        (model_path, details))
    print(f"Finished Model84 {arm} fold {fold}: component={score:.9f}", flush=True)
    del model, x, labels, sample_weights
    gc.collect()


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold, arms):
    plan = protocol["plans"][fold]
    pending = [arm for arm in arms if read_checkpoint(directory / f"{arm}_fold{fold}.npz",
        {"protocol_sha256": fingerprint, "fold": fold, "arm": arm}) is None]
    if not pending:
        print(f"Model84 fold {fold} already complete", flush=True)
        return
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    base = read_cache(PARENT20, f"aligned_25_fold{fold}_features", {
        "protocol_sha256": CONFIG["model20_protocol_sha256"], "rows": len(sy),
        "names": json.loads((PARENT20 / "protocol.json").read_text())["contract"]["names"][fold],
        "tasks": len(tasks)})
    white = white_source.cached_white(PARENT57 / f"aligned_25_fold{fold}_white.npy",
        CONFIG["model57_protocol_sha256"], len(sy))
    if white is None:
        raise ValueError("Missing Model57 synthetic cache")
    sx = np.concatenate((np.asarray(base), np.asarray(white)), axis=1)
    if sx.shape != (len(sy), 317):
        raise ValueError("Model84 synthetic alignment failed")
    del tasks, base, white
    for arm in pending:
        fit_arm(directory, protocol, fingerprint, layout, real, fold, sx, sy, st, arm)
    del sx, sy, st
    gc.collect()


def fold_scores(layout, value):
    return np.asarray([tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)])


def processed_signals(directory, fingerprint, layout, completed):
    parents = baseline_source.parent_state()
    _, baseline, _ = baseline_source.predictions(layout, parents)
    m26, m35, _ = incumbent.margins(layout)
    anti_delta = m35.astype(np.float64) - m26.astype(np.float64)
    d26, _, s26 = voter.frozen_study()
    original = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        original[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
    signals = {}
    for arm in CONFIG["arms"]:
        probability = original.copy()
        for fold in completed:
            held = np.flatnonzero(layout.row_fold == fold)
            saved = read_checkpoint(directory / f"{arm}_fold{fold}.npz",
                {"protocol_sha256": fingerprint, "fold": fold, "arm": arm})
            if saved is None:
                raise ValueError(f"Missing Model84 {arm} fold {fold}")
            validate_predictions(saved, held)
            probability[held] = saved["prediction"]
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted = expit(replaced + anti_delta)
        signals[arm] = apply_causal_transform(adjusted, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32)
    return np.asarray(baseline, dtype=np.float32), signals


def pilot_decision(directory, fingerprint, layout):
    baseline, signals = processed_signals(directory, fingerprint, layout, CONFIG["pilot_folds"])
    base_scores = fold_scores(layout, baseline)
    choices, gains = [], []
    for arm, signal in signals.items():
        for weight in CONFIG["blend_weights"]:
            value = ((1 - weight) * baseline + weight * signal).astype(np.float32)
            gain = float(fold_scores(layout, value)[CONFIG["pilot_folds"]].mean() -
                         base_scores[CONFIG["pilot_folds"]].mean())
            choices.append([arm, weight])
            gains.append(gain)
    selected = int(np.argmax(gains))
    decision = {"choices": choices, "pilot_gains": gains, "selected_index": selected,
        "selected_arm": choices[selected][0], "selected_weight": choices[selected][1],
        "continue_training": bool(gains[selected] >= CONFIG["pilot_minimum_gain"])}
    write_json(directory / "pilot_decision.json", decision)
    print(f"Model84 pilot gains: {dict(zip(map(str, choices), gains))}", flush=True)
    return decision


def final_report(directory, fingerprint, layout, decision):
    baseline, signals = processed_signals(directory, fingerprint, layout, range(5))
    weight = decision["selected_weight"]
    candidate = ((1 - weight) * baseline + weight * signals[decision["selected_arm"]]).astype(np.float32)
    base_scores, scores = fold_scores(layout, baseline), fold_scores(layout, candidate)
    gains = scores - base_scores
    confirmation = np.asarray(CONFIG["confirmation_folds"])
    scorers = []
    for fold in confirmation:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows]) for value in (baseline, candidate)]))
    rng, boot = np.random.default_rng(CONFIG["bootstrap_seed"]), []
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local = []
        for series, pair in scorers:
            multiplicity = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            weights = np.zeros(len(layout.inner_series_fold)); weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in pair])
        mean = np.asarray(local).mean(axis=0); boot.append(float(mean[1] - mean[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", {"gains": np.asarray(boot)},
                {"protocol_sha256": fingerprint, "selected_index": decision["selected_index"]})
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirmation) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows]) -
            tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
    confirm_gain, confirm_score = float(gains[confirmation].mean()), float(scores[confirmation].mean())
    full_score, wins = float(scores.mean()), int(np.sum(gains[confirmation] > 0))
    passed = bool(confirm_score >= CONFIG["target_tsauc"] and full_score >= CONFIG["target_tsauc"]
        and confirm_gain >= CONFIG["minimum_confirmation_gain"]
        and wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    summary = {"model": 84, "pilot": decision, "baseline_fold_scores": base_scores.tolist(),
        "candidate_fold_scores": scores.tolist(), "fold_gains": gains.tolist(),
        "confirmation_score": confirm_score, "full_oof_score": full_score,
        "confirmation_gain": confirm_gain, "confirmation_wins": wins,
        "bootstrap_lower_95": lower, "split_half_gains": split, "passes": passed,
        "deployment": "No packaging, upload, or cloud run performed."}
    write_json(directory / "summary.json", summary)
    result = ("# Model84: LightGBM capacity diversity\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base_scores[confirmation].mean():.6f} | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| {decision['selected_arm']} weight {weight:.3f} | {confirm_score:.6f} | {full_score:.6f} | "
        f"{confirm_gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Fold gains: {gains.tolist()}. Split-half gains: {split}.\n\n{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result); (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p62, parent, dependencies, d26, d39, d62, d67 = parent_state()
        directory, protocol, fingerprint = initialize(p62, parent, dependencies, d26, d39, d62, d67)
        layout = load_development_layout(ROOT)
        base = read_cache(PARENT20, "real_features", {"protocol_sha256": CONFIG["model20_protocol_sha256"],
            "rows": layout.row_count, "names": parent["contract"]["union_names"],
            "tasks": int(np.sum(layout.inner_series_fold >= 0))})
        white = white_source.cached_white(PARENT57 / "real_white.npy",
            CONFIG["model57_protocol_sha256"], layout.row_count)
        if white is None:
            raise ValueError("Missing Model57 real cache")
        real = white_source.CombinedFeatures(base, white)
        pool = load_real_pool(layout)
        for fold in CONFIG["pilot_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold, list(CONFIG["arms"]))
        decision = pilot_decision(directory, fingerprint, layout)
        if not decision["continue_training"]:
            result = ("# Model84: LightGBM capacity diversity\n\nNo choice met the prespecified "
                "+0.0001 pilot gate. Target 0.640 not reached. No cloud activity.\n")
            (directory / "RESULTS.md").write_text(result); (HERE / "RESULTS.md").write_text(result)
            print(result, flush=True); return
        for fold in CONFIG["confirmation_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold, [decision["selected_arm"]])
        final_report(directory, fingerprint, layout, decision)


if __name__ == "__main__":
    main()
