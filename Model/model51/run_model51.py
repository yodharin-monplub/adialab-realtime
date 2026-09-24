"""Evaluate causal, history-whitened integrated Bayes evidence on held series."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint
from models.model20.features import load_real_pool
from models.model39 import run_model39 as incumbent
from models.model51.evidence import score_task
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed Model39 protocol")
    files = [HERE / "run_model51.py", HERE / "evidence.py", HERE / "prespec.json",
        HERE / "readme.txt", d39 / "protocol.json", ROOT / "models/model10/sources.py",
        ROOT / "models/model20/features.py", ROOT / "submissions/submission16/detector/core/detectors3.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz", ROOT / "tsauc.py"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        dependencies={name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "pyarrow")}))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model51 protocol mismatch")
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
        raise ValueError("Changed Model51 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model51 input: {relative}")
    return directory, protocol, fingerprint


def validate_scores(saved, held):
    np.testing.assert_array_equal(saved["held_rows"], held)
    values = saved["prediction"]
    if values.shape != (len(held),) or values.dtype != np.float32:
        raise ValueError("Misaligned Model51 scores")
    if not np.isfinite(values).all() or np.any((values < -5) | (values > 15)):
        raise ValueError("Invalid Model51 score range")


def score_fold(directory, fingerprint, layout, pool, fold):
    path = directory / f"evidence_fold{fold}.npz"
    meta = dict(protocol_sha256=fingerprint, fold=fold, arm="whitened_integrated_bayes")
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, meta)
    if saved is not None:
        validate_scores(saved, held)
        return
    positions = np.flatnonzero(layout.inner_series_fold == fold)
    expected_owner = np.repeat(positions, layout.online_lengths[positions])
    if not np.array_equal(layout.series_position[held], expected_owner):
        raise ValueError("Model51 held series are not aligned by owner")
    prediction = np.empty(len(held), dtype=np.float32)
    started = time.monotonic()
    offset = 0
    tasks = ((int(position), *pool[int(position)]) for position in positions)
    print(f"Scoring Model51 fold {fold}: {len(positions)} complete unseen series", flush=True)
    with ProcessPoolExecutor(max_workers=CONFIG["feature_workers"],
            mp_context=multiprocessing.get_context("spawn")) as executor:
        for number, (position, values) in enumerate(executor.map(
                score_task, tasks, chunksize=1, buffersize=8), 1):
            if position != positions[number - 1] or len(values) != layout.online_lengths[position]:
                raise ValueError("Model51 worker order or series length mismatch")
            prediction[offset:offset + len(values)] = values
            offset += len(values)
            if number % 100 == 0 or number == len(positions):
                print(f"Model51 fold {fold}: {number}/{len(positions)} series", flush=True)
    if offset != len(held):
        raise ValueError("Model51 prediction row count mismatch")
    validate_scores(dict(prediction=prediction, held_rows=held), held)
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    details = directory / f"evidence_fold{fold}.fit.json"
    write_json(details, dict(evidence_tsauc=score, series=len(positions),
        rows=len(held), elapsed_seconds=time.monotonic() - started))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), meta, (details,))
    print(f"Finished Model51 fold {fold}: evidence TSAUC={score:.9f} in {time.monotonic()-started:.1f}s", flush=True)


def evidence_vector(directory, fingerprint, layout, completed_folds):
    values = np.zeros(layout.row_count, dtype=np.float32)
    for fold in completed_folds:
        held = np.flatnonzero(layout.row_fold == fold)
        saved = read_checkpoint(directory / f"evidence_fold{fold}.npz",
            dict(protocol_sha256=fingerprint, fold=fold, arm="whitened_integrated_bayes"))
        if saved is None:
            raise ValueError(f"Incomplete Model51 fold {fold}")
        validate_scores(saved, held)
        values[held] = saved["prediction"]
    return values


def score_matrix(layout, baseline, evidence):
    values = [baseline]
    values += [(baseline.astype(np.float64) + weight * evidence).astype(np.float32)
        for weight in CONFIG["blend_weights"]]
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])
    return values, scores


def pilot_decision(directory, fingerprint, layout, baseline):
    evidence = evidence_vector(directory, fingerprint, layout, CONFIG["pilot_folds"])
    _, scores = score_matrix(layout, baseline, evidence)
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
        raise ValueError("Changed Model51 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, baseline, decision):
    evidence = evidence_vector(directory, fingerprint, layout, range(5))
    values, scores = score_matrix(layout, baseline, evidence)
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
    summary = dict(model=51, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model51: history-whitened integrated Bayes evidence", "",
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| model39 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |",
        f"| evidence weight {decision['selected_weight']:.2f} | {confirmation_score:.6f} | {full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |",
        "", f"Pilot gains: {decision['pilot_gains']}.", f"Confirmation split-half gains: {split}.",
        "Exploratory held-series OOF; inherited Model39 is development-informed.",
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
        result = ("# Model51: history-whitened integrated Bayes evidence\n\n"
            f"Pilot gains: {dict(zip(CONFIG['blend_weights'], decision['pilot_gains']))}.\n"
            "No weight met the prespecified +0.0001 pilot gate; confirmation folds were not evaluated.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"evidence_fold{fold}.json").exists() for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, baseline, decision)
    else:
        print(f"Pilot selected Model51 weight {decision['selected_weight']:.2f}; confirmation scores pending", flush=True)
    return True


def run(directory, fingerprint):
    layout = load_development_layout(ROOT)
    pool = load_real_pool(layout)
    for fold in CONFIG["pilot_folds"]:
        score_fold(directory, fingerprint, layout, pool, fold)
    if not report(directory, fingerprint):
        return
    for fold in CONFIG["confirmation_folds"]:
        score_fold(directory, fingerprint, layout, pool, fold)
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
