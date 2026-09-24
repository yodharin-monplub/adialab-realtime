"""Confirmation-only evaluation of the frozen Model38 smoother."""
import numpy as np
from scipy.special import expit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model38.run_model38 import CONFIG, HERE, ROOT, frozen_study, margins
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


def report():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    values = margins(layout)
    pilot = np.asarray(CONFIG["pilot_folds"])
    confirmation = np.asarray(CONFIG["confirmation_folds"])
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], margin[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for margin in values])
    np.testing.assert_allclose(scores[0, pilot], CONFIG["pilot_model26_scores"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(scores[1, pilot], CONFIG["pilot_candidate_scores"], rtol=0, atol=1e-12)
    gain = float(scores[1, confirmation].mean() - scores[0, confirmation].mean())
    wins = int(np.sum(scores[1, confirmation] > scores[0, confirmation]))
    meta = dict(protocol_sha256=fingerprint,
        model26_protocol_sha256=CONFIG["model26_protocol_sha256"])
    saved = read_checkpoint(directory / "bootstrap.npz", meta)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in confirmation:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], margin[rows],
            layout.time_online[rows], layout.series_position[rows]) for margin in values]))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    completed = len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local = []
        for series, pair in scorers:
            multiplicity = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            if repeat < completed:
                continue
            weights = np.zeros(len(layout.inner_series_fold))
            weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in pair])
        if repeat < completed:
            continue
        mean = np.asarray(local).mean(axis=0)
        boot.append(float(mean[1] - mean[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)), meta)
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    split = []
    half = (layout.series_ids * 0x9E3779B1) & 1
    confirmation_rows = np.isin(layout.row_fold, confirmation)
    for value in (0, 1):
        rows = confirmation_rows & (half[layout.series_position] == value)
        split.append(float(tsauc(layout.target[rows], values[1][rows], layout.time_online[rows])
                           - tsauc(layout.target[rows], values[0][rows], layout.time_online[rows])))
    confirmation_mean = float(scores[1, confirmation].mean())
    passed = bool(confirmation_mean >= CONFIG["target_tsauc"]
        and gain >= CONFIG["minimum_confirmation_gain"]
        and wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    result = dict(model26=describe(scores[0, confirmation], scores[0, confirmation]),
        ema_smoother=describe(scores[1, confirmation], scores[0, confirmation]))
    result["ema_smoother"].update(fold_wins=wins, bootstrap_lower_95=lower,
        split_half_gains=split, passes=passed,
        no_break_false_alarms=false_alarms(expit(values[1].astype(np.float64)), layout, [.5, .8]))
    summary = dict(model=38, pilot_scores=dict(model26=scores[0, pilot].tolist(),
        ema_smoother=scores[1, pilot].tolist()), confirmation_scores=result,
        target_tsauc=CONFIG["target_tsauc"], target_reached=confirmation_mean >= CONFIG["target_tsauc"],
        selected_candidate="ema_smoother" if passed else None,
        interpretation="Transform selected on folds0-1 only; folds2-4 are untouched confirmation.",
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model38: pilot-selected short-memory smoothing", "",
        "| Arm | Confirmation TSAUC | Gain | Wins | Lower 95% | Split gains | Pass |",
        "|---|---:|---:|---:|---:|---|---|",
        f"| model26 | {scores[0, confirmation].mean():.6f} | +0.000000 | 0/3 | +0.000000 | baseline | baseline |",
        f"| ema_smoother | {confirmation_mean:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {split} | {passed} |",
        "", f"Pilot gain (folds0-1): {scores[1, pilot].mean()-scores[0, pilot].mean():+.6f}.",
        f"Confirmation target: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.",
        "", summary["interpretation"], "", "These are local held-series OOF scores, not cloud scores.",
        "", summary["deployment"], ""]
    text = "\n".join(lines)
    (directory / "RESULTS.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print(text, flush=True)
    return summary

