"""Frozen pilot and confirmation report for Model79."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model11.pipeline import write_json
from models.model18.run_model18 import commit_checkpoint
from models.model62 import report62
from models.model67 import run_model67
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
ARM = "lightgbm_aligned_40"


def new_prediction(layout, directory, fingerprint, folds):
    values = report62.candidate_margins(layout, directory, fingerprint, folds)
    if report62.CHOICES[3] != (0.0, 1.0):
        raise ValueError("Changed direct-voter report choice")
    return np.asarray(values[4], dtype=np.float32)


def candidate_values(layout, directory, fingerprint, folds):
    parents = run_model67.parent_state()
    selected, baseline, _ = run_model67.predictions(layout, parents)
    new = new_prediction(layout, directory, fingerprint, folds)
    values = [baseline]
    for fraction in CONFIG["replacement_fractions"]:
        values.append((baseline + 0.5 * fraction * (new - selected["62"])).astype(np.float32))
    return values


def score_folds(layout, values, folds):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in folds] for value in values])


def pilot_decision(directory, fingerprint, layout):
    folds = CONFIG["pilot_folds"]
    values = candidate_values(layout, directory, fingerprint, folds)
    scores = score_folds(layout, values, folds)
    gains = scores[1:].mean(axis=1) - scores[0].mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, fractions=CONFIG["replacement_fractions"],
        pilot_gains=gains.tolist(), baseline_scores=scores[0].tolist(),
        candidate_scores=scores[selected + 1].tolist(), selected_index=selected,
        selected_fraction=CONFIG["replacement_fractions"][selected],
        continue_training=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model79 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, decision):
    values = candidate_values(layout, directory, fingerprint, range(5))
    scores = score_folds(layout, values, range(5))
    base = scores[0]
    candidate = scores[decision["selected_index"] + 1]
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
                dict(protocol_sha256=fingerprint, selected_index=decision["selected_index"]))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], values[decision["selected_index"] + 1][rows],
            layout.time_online[rows]) - tsauc(layout.target[rows], values[0][rows],
            layout.time_online[rows])))
    confirmation_score = float(candidate[confirm].mean())
    full_score = float(candidate.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and full_score >= CONFIG["target_tsauc"]
        and gain >= CONFIG["minimum_confirmation_gain"] and wins >= CONFIG["minimum_confirmation_wins"]
        and lower > 0 and min(split) > 0)
    summary = dict(model=79, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model79: regime-matched aligned splices\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| replace {decision['selected_fraction']:.2f} of Model62 allocation | {confirmation_score:.6f} | "
        f"{full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {dict(zip(CONFIG['replacement_fractions'], decision['pilot_gains']))}.\n"
        f"Confirmation split-half gains: {split}.\n\n{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint, layout):
    decision = pilot_decision(directory, fingerprint, layout)
    if not decision["continue_training"]:
        summary = dict(model=79, pilot=decision, passes=False,
            deployment="No packaging, upload, or cloud run performed.")
        write_json(directory / "summary.json", summary)
        result = ("# Model79: regime-matched aligned splices\n\n"
            f"Pilot gains: {dict(zip(CONFIG['replacement_fractions'], decision['pilot_gains']))}.\n"
            "No choice met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"{ARM}_fold{fold}.npz").exists()
           for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, decision)
    else:
        print(f"Model79 pilot selected replacement fraction {decision['selected_fraction']:.2f}; "
              "confirmation fits pending", flush=True)
    return True
