"""Paired real-series evaluation; never trains or selects additional recipes."""
import json
from pathlib import Path

import numpy as np

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.run_model18 import (ARMS, HERE, ROOT, checkpoint_meta,
    commit_checkpoint, digest, read_checkpoint, validate_predictions)
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


def simultaneous_bounds(observed, bootstrap, confidence=.95):
    observed = np.asarray(observed)
    bootstrap = np.asarray(bootstrap)
    if bootstrap.ndim != 2 or bootstrap.shape[1:] != observed.shape or not np.isfinite(bootstrap).all():
        raise ValueError("Invalid joint bootstrap contrasts")
    critical = float(np.quantile(np.max(bootstrap-observed, axis=1), confidence))
    return observed-critical, critical


def choose_winner(results, config):
    passing = [name for name in ARMS[1:] if results[name]["mean_gain"] >= config["minimum_gain"]
        and results[name]["fold_wins"] >= config["minimum_fold_wins"]
        and results[name]["simultaneous_lower_95"] > 0]
    passing.sort(key=lambda name: (-results[name]["mean_gain"], int(name.split("_")[1]),
                                   not name.startswith("raw")))
    return passing[0] if passing else None


def false_alarms(prediction, layout, thresholds):
    no_break = np.flatnonzero((layout.inner_series_fold >= 0) & (layout.tau_index == -1))
    maxima = np.full(len(layout.inner_series_fold), -np.inf)
    np.maximum.at(maxima, layout.series_position, prediction)
    return {str(threshold): dict(series_count=len(no_break),
        crossing_count=int(np.sum(maxima[no_break] >= threshold)),
        crossing_rate=float(np.mean(maxima[no_break] >= threshold))) for threshold in thresholds}


def report(directory=None):
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"] if directory is None else Path(directory)
    protocol_path = directory / "protocol.json"
    protocol_sha = sha256(protocol_path)
    if directory == ROOT / latest["directory"] and protocol_sha != latest["protocol_sha256"]:
        raise ValueError("Frozen protocol checksum mismatch")
    protocol = json.loads(protocol_path.read_text())
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Frozen donor plans corrupted")
    config = protocol["contract"]["config"]
    if config["arms"] != list(ARMS):
        raise ValueError("Report arm order differs from frozen protocol")
    # Verify label/row alignment inputs; report never opens the outer labels.
    for relative in ("train/y_train_index.parquet", "models/model5/artifacts/split_assignments.npz"):
        key = "target_index_sha256" if "y_train" in relative else "split_sha256"
        if sha256(ROOT / relative) != protocol["contract"]["features"][key]:
            raise ValueError("Evaluation input changed since freezing")
    metadata_path = "models/artifacts/core_features.metadata.npz"
    if sha256(ROOT / metadata_path) != protocol["contract"]["file_hashes"][metadata_path]:
        raise ValueError("Evaluation row metadata changed since freezing")
    layout = load_development_layout(ROOT)
    predictions, fold_scores = [], []
    for arm in ARMS:
        prediction = np.empty(layout.row_count, dtype=np.float32)
        scores = []
        for fold in range(5):
            held = np.flatnonzero(layout.row_fold == fold)
            saved = read_checkpoint(directory / f"{arm}_fold{fold}.npz", checkpoint_meta(protocol_sha, fold, arm))
            if saved is None:
                raise RuntimeError(f"Incomplete study: missing {arm} fold {fold}; run resumes fitting")
            validate_predictions(saved, held)
            prediction[held] = saved["prediction"]
            scores.append(tsauc(layout.target[held], saved["prediction"], layout.time_online[held]))
        predictions.append(prediction)
        fold_scores.append(scores)
    np.testing.assert_allclose(fold_scores[0], protocol["contract"]["expected_control_scores"],
                               atol=config["control_tolerance"], rtol=0)
    observed = np.mean(fold_scores, axis=1)[1:] - np.mean(fold_scores[0])
    checkpoint = directory / "bootstrap.npz"
    meta = dict(protocol_sha256=protocol_sha, prediction_hashes=[
        sha256(directory / f"{arm}_fold{fold}.npz") for arm in ARMS for fold in range(5)])
    saved = read_checkpoint(checkpoint, meta)
    differences = [] if saved is None else saved["differences"].tolist()
    if len(differences) > config["bootstrap_replicates"] or (differences and np.asarray(differences).shape != (len(differences), 4)):
        raise ValueError("Invalid bootstrap checkpoint length/shape")
    if len(differences) < config["bootstrap_replicates"]:
        scorers, fold_series = [], []
        for fold in range(5):
            rows = layout.row_fold == fold
            fold_series.append(np.flatnonzero(layout.inner_series_fold == fold))
            scorers.append([WeightedTSAUC(layout.target[rows], p[rows], layout.time_online[rows],
                                          layout.series_position[rows]) for p in predictions])
            np.testing.assert_allclose([s.score(np.ones(len(layout.inner_series_fold))) for s in scorers[-1]],
                                       np.asarray(fold_scores)[:, fold], atol=1e-12, rtol=0)
        rng = np.random.default_rng(config["bootstrap_seed"])
        completed = len(differences)
        for repeat in range(config["bootstrap_replicates"]):
            scores = []
            for series, local in zip(fold_series, scorers):
                # Advance the original RNG even when resuming past completed draws.
                multiplicity = rng.multinomial(len(series), np.full(len(series), 1/len(series)))
                if repeat < completed:
                    continue
                weight = np.zeros(len(layout.inner_series_fold))
                weight[series] = multiplicity
                scores.append([scorer.score(weight) for scorer in local])
            if repeat < completed:
                continue
            mean = np.mean(scores, axis=0)
            differences.append((mean[1:] - mean[0]).tolist())
            if (repeat+1) % 25 == 0 or repeat+1 == config["bootstrap_replicates"]:
                commit_checkpoint(checkpoint, dict(differences=np.asarray(differences)), meta)
                print(f"Paired whole-series bootstrap: {repeat+1}/{config['bootstrap_replicates']}", flush=True)
    lower, critical = simultaneous_bounds(observed, differences, config["simultaneous_confidence"])
    results = {}
    for index, arm in enumerate(ARMS):
        result = describe(fold_scores[index], np.asarray(fold_scores[0]))
        result["fold_gains"] = (np.asarray(fold_scores[index])-fold_scores[0]).tolist()
        result["no_break_false_alarms"] = false_alarms(predictions[index], layout, config["false_alarm_thresholds"])
        result["actual_synthetic_weight_shares"] = [json.loads(
            (directory / f"{arm}_fold{fold}.fit.json").read_text())["weights"]["actual_share"] for fold in range(5)]
        if index:
            result["simultaneous_lower_95"] = float(lower[index-1])
            result["paired_bootstrap_standard_error"] = float(np.std(np.asarray(differences)[:, index-1], ddof=1))
        results[arm] = result
    winner = choose_winner(results, config)
    summary = dict(model=18, run_directory=str(directory.relative_to(ROOT)), results=results,
        selected_standalone_candidate=winner, bootstrap_replicates=len(differences),
        simultaneous_critical_value=critical, donor_stats=[p["stats"] for p in protocol["plans"]],
        control_maximum_score_error=float(np.max(np.abs(np.asarray(fold_scores[0])-protocol["contract"]["expected_control_scores"]))),
        interpretation="Reused development folds only. Paired whole-series bootstrap conditions on saved OOF predictions, not upstream research or training uncertainty. Synthetic labels are assumptions, not guaranteed detectable breaks.",
        reserved_holdout="Historically revealed; not reopened", competition_test_labels_loaded=False,
        deployment="Model 16 unchanged; no ensemble, cloud submission, or full-data promotion.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model 18: training-only splice augmentation", "",
        "Five whole-series development folds; fresh Model 11 control. Higher TSAUC is better.", "",
        "| Arm | Mean TSAUC | Gain | Fold wins | Simultaneous 95% lower gain |",
        "|---|---:|---:|---:|---:|"]
    for arm, result in results.items():
        bound = "—" if arm == "control" else f'{result["simultaneous_lower_95"]:+.6f}'
        lines.append(f'| {arm} | {result["fold_mean"]:.6f} | {result["mean_gain"]:+.6f} | {result["fold_wins"]}/5 | {bound} |')
    lines += ["", "Pass requires gain >= 0.001, at least four fold wins, and simultaneous lower bound > 0.",
        f"Selected standalone candidate: {winner or 'none; augmentation did not pass the fixed gates'}.", "",
        "## Fold TSAUC", "", "| Arm | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, result in results.items():
        lines.append(f"| {arm} | " + " | ".join(f"{s:.6f}" for s in result["fold_scores"]) + " |")
    lines += ["", "## Real no-break false alarms", "",
        "Fraction of complete real no-break series with any online prediction >= the fixed threshold; not calibrated probabilities.",
        "", "| Arm | >= 0.5 | >= 0.8 |", "|---|---:|---:|"]
    for arm, result in results.items():
        rates = result["no_break_false_alarms"]
        lines.append(f'| {arm} | {rates["0.5"]["crossing_rate"]:.2%} | {rates["0.8"]["crossing_rate"]:.2%} |')
    lines += ["", summary["interpretation"], "", "The historically revealed outer holdout remained closed. No competition test labels were used.",
        "", summary["deployment"], "", f"Run artifacts: `{summary['run_directory']}`.",
        "Full paired fold gains, donor shortfalls, weight accounting, and counts are in `summary.json`.", ""]
    text = "\n".join(lines)
    (directory / "RESULTS.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print(text, flush=True)
    return summary
