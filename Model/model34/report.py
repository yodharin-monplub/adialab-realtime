"""Held-series report for the Model34 pairwise residual correction."""
import numpy as np
from scipy.special import logit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model34.run_model34 import (
    ARMS, CONFIG, HERE, ROOT, frozen_study, level_one_and_base, prediction_checkpoint,
)
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


EXPECTED_MODEL26 = np.asarray([
    .6427355043536926, .6085023046201091, .6422531072481089,
    .6218047904181766, .6467391865285458,
])


def report():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    _, base_margin = level_one_and_base(layout)
    candidate = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        candidate[held] = prediction_checkpoint(directory, fold, fingerprint, held)
    margins = [base_margin, logit(np.clip(candidate, 1e-7, 1 - 1e-7))]
    scores = np.asarray([
        [tsauc(layout.target[layout.row_fold == fold], margin[layout.row_fold == fold],
               layout.time_online[layout.row_fold == fold]) for fold in range(5)]
        for margin in margins
    ])
    np.testing.assert_allclose(scores[0], EXPECTED_MODEL26, rtol=0, atol=1e-12)
    gain = float(scores[1].mean() - scores[0].mean())
    wins = int(np.sum(scores[1] > scores[0]))
    half = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for value in (0, 1):
        rows = half[layout.series_position] == value
        split.append(float(tsauc(layout.target[rows], margins[1][rows], layout.time_online[rows])
                           - tsauc(layout.target[rows], margins[0][rows], layout.time_online[rows])))
    hashes = {f"residual_{fold}": sha256(directory / f"residual_fold{fold}.npz") for fold in range(5)}
    metadata = dict(protocol_sha256=fingerprint, predictions=hashes)
    saved = read_checkpoint(directory / "bootstrap.npz", metadata)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in range(5):
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], margin[rows],
            layout.time_online[rows], layout.series_position[rows]) for margin in margins]))
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
        values = np.asarray(local).mean(axis=0)
        boot.append(float(values[1] - values[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)), metadata)
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    passed = bool(scores[1].mean() >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_gain"]
                  and wins >= CONFIG["minimum_fold_wins"] and lower > 0 and min(split) > 0)
    results = {arm: describe(scores[index], scores[0]) for index, arm in enumerate(ARMS)}
    results["pairwise_residual"].update(
        fold_wins=wins, bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        no_break_false_alarms=false_alarms(candidate.astype(np.float64), layout, [.5, .8]),
    )
    summary = dict(
        model=34,
        selected_candidate="pairwise_residual" if passed else None,
        target_tsauc=CONFIG["target_tsauc"],
        target_reached=bool(scores[1].mean() >= CONFIG["target_tsauc"]),
        results=results,
        interpretation="Model26 is the immutable initial margin; trees learn only a same-time pairwise additive correction.",
        deployment="No packaging, upload, or cloud run performed.",
    )
    write_json(directory / "summary.json", summary)
    lines = ["# Model34: TSAUC pairwise residual correction", "",
        "| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |",
        "|---|---:|---:|---:|---:|---|---|"]
    for arm in ARMS:
        result = results[arm]
        lines.append(f"| {arm} | {result['fold_mean']:.6f} | {result['mean_gain']:+.6f} | "
            f"{result.get('fold_wins', 0)}/5 | {result.get('bootstrap_lower_95', 0):+.6f} | "
            f"{result.get('split_half_gains', 'baseline')} | {result.get('passes', 'baseline')} |")
    lines += ["", f"Target held-series OOF TSAUC: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.",
        "", summary["interpretation"], "", "These are local development OOF scores, not cloud scores.",
        "", summary["deployment"], ""]
    text = "\n".join(lines)
    (directory / "RESULTS.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print(text, flush=True)
    return summary
