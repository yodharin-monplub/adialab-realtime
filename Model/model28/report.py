"""Confirmation-only OOF report for Model28."""
import json

import numpy as np
from scipy.special import expit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model20.report import ensemble_margin
from models.model28.run_model28 import ROOT, HERE, ARMS, CONFIG, frozen_study, prediction_checkpoint
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


def report():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    component, margins = [], []
    for arm in ARMS:
        probability = np.empty(layout.row_count, dtype=np.float32)
        for fold in range(5):
            held = np.flatnonzero(layout.row_fold == fold)
            probability[held] = prediction_checkpoint(arm, fold, held)
        component.append(probability)
        margins.append(ensemble_margin(probability, layout))
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], margin[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for margin in margins])
    component_scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], probability[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for probability in component])
    confirm = np.asarray(CONFIG["confirmation_folds"], dtype=np.int64)
    gain = float(scores[1, confirm].mean() - scores[0, confirm].mean())
    wins = int(np.sum(scores[1, confirm] > scores[0, confirm]))
    half = (layout.series_ids * 0x9E3779B1) & 1
    split_gains = []
    for value in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (half[layout.series_position] == value)
        split_gains.append(float(tsauc(layout.target[rows], margins[1][rows], layout.time_online[rows])
            - tsauc(layout.target[rows], margins[0][rows], layout.time_online[rows])))
    prediction_hashes = {f"model27_fold{fold}": sha256(
        (ROOT / json.loads((ROOT / "models/model27/runs/latest.json").read_text())["directory"])
        / f"matched_context_fold{fold}.npz") for fold in range(5)}
    meta = dict(protocol_sha256=fingerprint, predictions=prediction_hashes)
    saved = read_checkpoint(directory / "bootstrap.npz", meta)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in confirm:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], margin[rows],
            layout.time_online[rows], layout.series_position[rows]) for margin in margins]))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    completed = len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local = []
        for series, fold_scorers in scorers:
            multiplicity = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            if repeat < completed:
                continue
            weights = np.zeros(len(layout.inner_series_fold))
            weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in fold_scorers])
        if repeat < completed:
            continue
        local = np.asarray(local).mean(axis=0)
        boot.append(float(local[1] - local[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)), meta)
    lower = float(np.quantile(np.asarray(boot), (1 - CONFIG["confidence"]) / 2))
    confirmation_score = float(scores[1, confirm].mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"]
        and gain >= CONFIG["minimum_confirmation_gain"]
        and wins >= CONFIG["minimum_confirmation_wins"]
        and lower > 0 and min(split_gains) > 0)
    results = {}
    for index, arm in enumerate(ARMS):
        results[arm] = dict(
            five_fold=describe(scores[index], scores[0]),
            component=describe(component_scores[index], component_scores[0]),
            pilot_score=float(scores[index, CONFIG["pilot_fold"]]),
            confirmation_scores=scores[index, confirm].tolist(),
            confirmation_mean=float(scores[index, confirm].mean()),
            no_break_false_alarms=false_alarms(expit(margins[index].astype(np.float64)), layout, [.5, .8]),
        )
    results["pilot_80"].update(confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_confirmation_gains=split_gains, passes=passed)
    summary = dict(model=28, selected_candidate="pilot_80" if passed else None,
        target_tsauc=CONFIG["target_tsauc"], target_reached=bool(confirmation_score >= CONFIG["target_tsauc"]),
        results=results, interpretation=CONFIG["interpretation"],
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model28: pilot-selected 80% context blend", "",
        "| Arm | Pilot fold 0 | Confirmation folds 1-4 | Confirmation gain | Wins | Lower 95% | Pass |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        result = results[arm]
        lines.append(f"| {arm} | {result['pilot_score']:.6f} | {result['confirmation_mean']:.6f} | "
            f"{result.get('confirmation_gain', 0):+.6f} | {result.get('confirmation_wins', 0)}/4 | "
            f"{result.get('bootstrap_lower_95', 0):+.6f} | {result.get('passes', 'baseline')} |")
    lines += ["", f"Unseen confirmation target: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.",
        "Fold 0 is excluded from confirmation because it selected the blend weight.", "", CONFIG["interpretation"],
        "", "These are local held-series OOF scores, not cloud scores.", "", summary["deployment"], ""]
    rendered = "\n".join(lines)
    (directory / "RESULTS.md").write_text(rendered)
    (HERE / "RESULTS.md").write_text(rendered)
    print(rendered, flush=True)
    return summary


if __name__ == "__main__":
    report()
