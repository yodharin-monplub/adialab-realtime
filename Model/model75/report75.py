"""Pilot and confirmation report for Model75's full-row GPU voter."""
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
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, read_checkpoint, validate_predictions
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.model67 import run_model67 as best_parent
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
CHOICES = list(itertools.product(CONFIG["voter_weights"], CONFIG["ensemble_weights"]))


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model75 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed Model75 input: {relative}")
    prepared = json.loads((directory / "prepared.json").read_text())
    if prepared["protocol_sha256"] != fingerprint:
        raise ValueError("Changed Model75 preparation")
    for item in [prepared["layout"], *prepared["synthetic"]]:
        if sha256(directory / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed prepared Model75 artifact: {item['path']}")
    return directory, protocol, fingerprint


def completed_folds(directory):
    return [fold for fold in range(5)
        if (directory / f"{CONFIG['arm']}_fold{fold}.json").exists()]


def values(layout, directory, fingerprint, completed):
    m26, m35, _ = incumbent.margins(layout)
    anti_delta = m35.astype(np.float64) - m26.astype(np.float64)
    d26, _, s26 = voter.frozen_study()
    original = np.empty(layout.row_count, dtype=np.float32)
    replacement = np.empty_like(original)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        original[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        if fold in completed:
            saved = read_checkpoint(directory / f"{CONFIG['arm']}_fold{fold}.npz", {
                "protocol_sha256": fingerprint, "fold": fold, "arm": CONFIG["arm"]})
            if saved is None:
                raise ValueError(f"Incomplete Model75 fold {fold}")
            validate_predictions(saved, held)
            replacement[held] = saved["prediction"]
        else:
            replacement[held] = original[held]
    parents = best_parent.parent_state()
    _, baseline, _ = best_parent.predictions(layout, parents)
    candidates = []
    for voter_weight, ensemble_weight in CHOICES:
        probability = ((1 - voter_weight) * original + voter_weight * replacement).astype(np.float32)
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted = expit(replaced + anti_delta)
        final = apply_causal_transform(adjusted, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32)
        candidates.append(((1 - ensemble_weight) * baseline + ensemble_weight * final).astype(np.float32))
    return np.asarray(baseline, dtype=np.float32), candidates


def fold_scores(layout, predictions):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in predictions])


def pilot(directory, fingerprint, layout, scores):
    folds = np.asarray(CONFIG["pilot_folds"])
    gains = scores[1:, folds].mean(axis=1) - scores[0, folds].mean()
    selected = int(np.argmax(gains))
    decision = {
        "protocol_sha256": fingerprint,
        "choices": [list(choice) for choice in CHOICES],
        "pilot_gains": gains.tolist(),
        "baseline_scores": scores[0, folds].tolist(),
        "candidate_scores": scores[selected + 1, folds].tolist(),
        "selected_index": selected,
        "selected_voter_weight": CHOICES[selected][0],
        "selected_ensemble_weight": CHOICES[selected][1],
        "continue_training": bool(gains[selected] >= CONFIG["pilot_minimum_gain"]),
    }
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model75 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, baseline, candidate, base_scores, scores, decision):
    confirm = np.asarray(CONFIG["confirmation_folds"])
    gain = float(scores[confirm].mean() - base_scores[confirm].mean())
    wins = int(np.sum(scores[confirm] > base_scores[confirm]))
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
            commit_checkpoint(directory / "bootstrap.npz", {"gains": np.asarray(boot)}, {
                "protocol_sha256": fingerprint, "selected_index": decision["selected_index"]})
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows]) -
            tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
    confirmation_score, full_score = float(scores[confirm].mean()), float(scores.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and
        full_score >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_confirmation_gain"] and
        wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    summary = {
        "model": 75, "pilot": decision,
        "baseline_fold_scores": base_scores.tolist(), "candidate_fold_scores": scores.tolist(),
        "confirmation_score": confirmation_score, "full_oof_score": full_score,
        "confirmation_gain": gain, "confirmation_wins": wins,
        "bootstrap_lower_95": lower, "split_half_gains": split, "passes": passed,
        "deployment": "No packaging, upload, or CrunchDAO cloud run performed.",
    }
    write_json(directory / "summary.json", summary)
    result = (
        "# Model75: full-row GPU XGBoost voter\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base_scores[confirm].mean():.6f} | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| voter {decision['selected_voter_weight']:.2f}, ensemble {decision['selected_ensemble_weight']:.2f} | "
        f"{confirmation_score:.6f} | {full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {dict(zip(map(str, CHOICES), decision['pilot_gains']))}.\n"
        f"Confirmation split-half gains: {split}.\n\n{summary['deployment']}\n"
    )
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    completed = completed_folds(directory)
    if not set(CONFIG["pilot_folds"]).issubset(completed):
        print(f"Model75 pilot folds pending; completed={completed}", flush=True)
        return
    baseline, candidates = values(layout, directory, fingerprint, completed)
    all_scores = fold_scores(layout, [baseline, *candidates])
    decision = pilot(directory, fingerprint, layout, all_scores)
    if not decision["continue_training"]:
        result = (
            "# Model75: full-row GPU XGBoost voter\n\n"
            f"Pilot gains: {dict(zip(map(str, CHOICES), decision['pilot_gains']))}.\n"
            "No choice met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n"
        )
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return
    if not set(CONFIG["confirmation_folds"]).issubset(completed):
        print(f"Model75 pilot passed with voter={decision['selected_voter_weight']}, "
            f"ensemble={decision['selected_ensemble_weight']}; confirmation folds pending", flush=True)
        return
    selected = candidates[decision["selected_index"]]
    final_scores = all_scores[decision["selected_index"] + 1]
    final_report(directory, fingerprint, layout, baseline, selected,
        all_scores[0], final_scores, decision)


if __name__ == "__main__":
    main()
