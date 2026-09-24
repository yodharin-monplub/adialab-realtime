"""Frozen pilot and confirmation for 60%-mass LightGBM versus Model62."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.causal_transforms import apply_causal_transform
from models.model11.pipeline import write_json
from models.model18.run_model18 import commit_checkpoint, read_checkpoint, validate_predictions
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.model62 import report62 as source_report
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT62 = ROOT / CONFIG["model62_run"]
NEW_ARM = "lightgbm_aligned_60"
OLD_ARM = "lightgbm_aligned_40"
CHOICES = list(itertools.product(CONFIG["old_fraction"], CONFIG["blend_weights"]))


def candidate_margins(layout, directory, fingerprint, completed_folds):
    m26, m35, m39 = incumbent.margins(layout)
    old_summary = json.loads((PARENT62 / "summary.json").read_text())
    old_selected = old_summary["pilot"]["selected_index"]
    old_values = source_report.candidate_margins(layout, PARENT62,
        CONFIG["model62_protocol_sha256"], range(5))
    baseline = old_values[old_selected + 1]
    del old_values
    anti_delta = m35.astype(np.float64) - m26.astype(np.float64)
    d26, _, s26 = voter.frozen_study()
    original = np.empty(layout.row_count, dtype=np.float32)
    new = np.empty_like(original)
    old = np.empty_like(original)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        original[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        prior = read_checkpoint(PARENT62 / f"{OLD_ARM}_fold{fold}.npz",
            dict(protocol_sha256=CONFIG["model62_protocol_sha256"], fold=fold, arm=OLD_ARM))
        if prior is None:
            raise ValueError(f"Incomplete Model62 fold {fold}")
        validate_predictions(prior, held)
        old[held] = prior["prediction"]
        if fold in completed_folds:
            saved = read_checkpoint(directory / f"{NEW_ARM}_fold{fold}.npz",
                dict(protocol_sha256=fingerprint, fold=fold, arm=NEW_ARM))
            if saved is None:
                raise ValueError(f"Incomplete Model65 fold {fold}")
            validate_predictions(saved, held)
            new[held] = saved["prediction"]
        else:
            new[held] = old[held]
    values = [baseline]
    for old_fraction, weight in CHOICES:
        mixed = ((1 - old_fraction) * new + old_fraction * old).astype(np.float32)
        probability = ((1 - weight) * original + weight * mixed).astype(np.float32)
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted = expit(replaced + anti_delta)
        values.append(apply_causal_transform(adjusted, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32))
    return values, m39


def fold_scores(layout, values):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])


def pilot_decision(directory, fingerprint, layout):
    values, _ = candidate_margins(layout, directory, fingerprint, CONFIG["pilot_folds"])
    scores = fold_scores(layout, values)
    pilot = np.asarray(CONFIG["pilot_folds"])
    gains = scores[1:, pilot].mean(axis=1) - scores[0, pilot].mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, choices=[list(pair) for pair in CHOICES],
        pilot_gains=gains.tolist(), baseline_scores=scores[0, pilot].tolist(),
        candidate_scores=scores[selected + 1, pilot].tolist(),
        selected_index=selected, selected_old_fraction=CHOICES[selected][0],
        selected_weight=CHOICES[selected][1],
        continue_training=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model65 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, decision):
    values, model39 = candidate_margins(layout, directory, fingerprint, range(5))
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
                dict(protocol_sha256=fingerprint, selected_index=decision["selected_index"]))
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
    m39_scores = fold_scores(layout, [model39])[0]
    summary = dict(model=65, pilot=decision, model39_fold_scores=m39_scores.tolist(),
        baseline_fold_scores=base.tolist(), candidate_fold_scores=candidate.tolist(),
        confirmation_score=confirmation_score, full_oof_score=full_score,
        confirmation_gain=gain, confirmation_wins=wins, bootstrap_lower_95=lower,
        split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model65: 60%-synthetic LightGBM voter\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain vs Model62 | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| model39 | {m39_scores[confirm].mean():.6f} | {m39_scores.mean():.6f} | — | — | — | — |\n"
        f"| model62 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| 60%-mass, old fraction {decision['selected_old_fraction']:.2f}, voter weight "
        f"{decision['selected_weight']:.2f} | {confirmation_score:.6f} | {full_score:.6f} | "
        f"{gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains vs Model62: {dict(zip(map(str, CHOICES), decision['pilot_gains']))}.\n"
        f"Confirmation split-half gains vs Model62: {split}.\n"
        "Exploratory held-series OOF; inherited features and ensemble are development-informed.\n\n"
        f"{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint, layout):
    decision = pilot_decision(directory, fingerprint, layout)
    if not decision["continue_training"]:
        result = ("# Model65: 60%-synthetic LightGBM voter\n\n"
            f"Pilot gains vs Model62: {dict(zip(map(str, CHOICES), decision['pilot_gains']))}.\n"
            "No choice met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"{NEW_ARM}_fold{fold}.npz").exists()
           for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, decision)
    else:
        print(f"Model65 pilot selected old fraction {decision['selected_old_fraction']:.2f}, "
            f"voter weight {decision['selected_weight']:.2f}; confirmation fits pending", flush=True)
    return True
