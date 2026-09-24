"""Evaluate frozen within-time rank aggregation of Model67's parents."""
from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest
from models.model67 import run_model67
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
CHOICES = list(itertools.product(CONFIG["aggregation_methods"],
                                 CONFIG["replacement_weights"]))


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    directory67, _, fingerprint67 = run_model67.frozen_study()
    if (str(directory67.relative_to(ROOT)) != CONFIG["model67_run"] or
            fingerprint67 != CONFIG["model67_protocol_sha256"]):
        raise ValueError("Changed Model67 parent")
    paths = [HERE / "run_model76.py", HERE / "prespec.json", HERE / "readme.txt",
             directory67 / "protocol.json", directory67 / "summary.json",
             ROOT / "models/model67/run_model67.py", ROOT / "tsauc.py",
             ROOT / "models/model5/artifacts/split_assignments.npz",
             ROOT / "train/y_train_index.parquet"]
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths})
    protocol = dict(contract=contract)
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model76 protocol mismatch")
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
        raise ValueError("Changed Model76 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model76 input: {relative}")
    return directory, protocol, fingerprint


def within_time_rank(values, time_online):
    values = np.asarray(values)
    time_online = np.asarray(time_online)
    result = np.empty(values.shape, dtype=np.float32)
    for step in np.unique(time_online):
        rows = np.flatnonzero(time_online == step)
        local = values[rows]
        order = np.argsort(local, kind="mergesort")
        sorted_values = local[order]
        starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
        ends = np.r_[starts[1:], len(rows)]
        ranks = np.empty(len(rows), dtype=np.float64)
        ranks[order] = np.repeat(0.5 * (starts + ends - 1), ends - starts)
        result[rows] = ((ranks + 0.5) / len(rows)).astype(np.float32)
    return result


def candidates(layout):
    parents = run_model67.parent_state()
    selected, baseline, _ = run_model67.predictions(layout, parents)
    if CONFIG["parent_weights"] != run_model67.CONFIG["weights"]:
        raise ValueError("Model76 parent weights differ from Model67")
    ranked = {name: within_time_rank(value, layout.time_online)
              for name, value in selected.items()}
    ranked_baseline = within_time_rank(baseline, layout.time_online)
    borda = sum((CONFIG["parent_weights"][name] * ranked[name]
                 for name in sorted(ranked)), np.zeros(layout.row_count, dtype=np.float64))
    eps = 0.5 / max(1, layout.row_count)
    normal_margin = sum((CONFIG["parent_weights"][name] *
                         ndtri(np.clip(ranked[name], eps, 1 - eps))
                         for name in sorted(ranked)),
                        np.zeros(layout.row_count, dtype=np.float64))
    aggregate = {
        "borda": within_time_rank(borda, layout.time_online),
        "normal": within_time_rank(normal_margin, layout.time_online),
    }
    values = [baseline]
    for method, weight in CHOICES:
        values.append(((1 - weight) * ranked_baseline +
                       weight * aggregate[method]).astype(np.float32))
    return values


def fold_scores(layout, values):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])


def report(directory, fingerprint, layout):
    values = candidates(layout)
    scores = fold_scores(layout, values)
    pilot = np.asarray(CONFIG["pilot_folds"])
    pilot_gains = scores[1:, pilot].mean(axis=1) - scores[0, pilot].mean()
    selected_index = int(np.argmax(pilot_gains))
    method, weight = CHOICES[selected_index]
    decision = dict(protocol_sha256=fingerprint, choices=[list(x) for x in CHOICES],
        pilot_gains=pilot_gains.tolist(), selected_index=selected_index,
        selected_method=method, selected_weight=weight,
        continue_evaluation=bool(pilot_gains[selected_index] >= CONFIG["pilot_minimum_gain"]))
    write_json(directory / "pilot_decision.json", decision)
    if not decision["continue_evaluation"]:
        summary = dict(model=76, pilot=decision, passes=False,
            deployment="No packaging, upload, or cloud run performed.")
        write_json(directory / "summary.json", summary)
        result = ("# Model76: within-time rank aggregation\n\n"
            f"Pilot gains: {dict(zip(map(str, CHOICES), pilot_gains.tolist()))}.\n"
            "No choice met the prespecified +0.0001 pilot gate.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return

    base = scores[0]
    candidate = scores[selected_index + 1]
    confirm = np.asarray(CONFIG["confirmation_folds"])
    gain = float(candidate[confirm].mean() - base[confirm].mean())
    wins = int(np.sum(candidate[confirm] > base[confirm]))
    scorers = []
    for fold in confirm:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        pair = [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows])
            for value in (values[0], values[selected_index + 1])]
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
                dict(protocol_sha256=fingerprint, selected_index=selected_index))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], values[selected_index + 1][rows],
            layout.time_online[rows]) - tsauc(layout.target[rows], values[0][rows],
            layout.time_online[rows])))
    confirmation_score = float(candidate[confirm].mean())
    full_score = float(candidate.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and
        full_score >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_confirmation_gain"] and
        wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    summary = dict(model=76, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model76: within-time rank aggregation\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| {method}, weight {weight:.2f} | {confirmation_score:.6f} | {full_score:.6f} | "
        f"{gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {dict(zip(map(str, CHOICES), pilot_gains.tolist()))}.\n"
        f"Confirmation split-half gains: {split}.\n\n"
        f"{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, _, fingerprint = initialize()
        layout = load_development_layout(ROOT)
        report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
