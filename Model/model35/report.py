"""Evaluation of the adaptive Model35 post-process."""
import numpy as np
from scipy.special import expit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model35.run_model35 import ARMS, CONFIG, HERE, ROOT, frozen_study, margins
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


EXPECTED_MODEL26 = np.asarray([.6427355043536926, .6085023046201091, .6422531072481089,
                               .6218047904181766, .6467391865285458])


def report():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    values = margins(layout)
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], margin[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for margin in values])
    np.testing.assert_allclose(scores[0], EXPECTED_MODEL26, rtol=0, atol=1e-12)
    gain = float(scores[1].mean() - scores[0].mean())
    wins = int(np.sum(scores[1] > scores[0]))
    half = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for value in (0, 1):
        rows = half[layout.series_position] == value
        split.append(float(tsauc(layout.target[rows], values[1][rows], layout.time_online[rows])
                           - tsauc(layout.target[rows], values[0][rows], layout.time_online[rows])))
    source_hashes = {}
    for model, stem in ((30, "matched_15"), (34, "residual")):
        latest = __import__("json").loads((ROOT / f"models/model{model}/runs/latest.json").read_text())
        source = ROOT / latest["directory"]
        source_hashes.update({f"m{model}_{fold}": sha256(source / f"{stem}_fold{fold}.npz") for fold in range(5)})
    metadata = dict(protocol_sha256=fingerprint, predictions=source_hashes)
    saved = read_checkpoint(directory / "bootstrap.npz", metadata)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in range(5):
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
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)), metadata)
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    passed = bool(scores[1].mean() >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_gain"]
                  and wins >= CONFIG["minimum_fold_wins"] and lower > 0 and min(split) > 0)
    results = {arm: describe(scores[index], scores[0]) for index, arm in enumerate(ARMS)}
    results["anti_residual"].update(fold_wins=wins, bootstrap_lower_95=lower,
        split_half_gains=split, passes=passed,
        no_break_false_alarms=false_alarms(expit(values[1].astype(np.float64)), layout, [.5, .8]))
    summary = dict(model=35, selected_candidate="anti_residual" if passed else None,
        target_tsauc=CONFIG["target_tsauc"], target_reached=bool(scores[1].mean() >= CONFIG["target_tsauc"]),
        results=results, adaptive_selection=CONFIG["selection_status"],
        interpretation="Exploratory anti-ensemble removes redundant correction directions from Model26.",
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model35: exploratory anti-residual post-process", "",
        "| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |",
        "|---|---:|---:|---:|---:|---|---|"]
    for arm in ARMS:
        result = results[arm]
        lines.append(f"| {arm} | {result['fold_mean']:.6f} | {result['mean_gain']:+.6f} | "
            f"{result.get('fold_wins', 0)}/5 | {result.get('bootstrap_lower_95', 0):+.6f} | "
            f"{result.get('split_half_gains', 'baseline')} | {result.get('passes', 'baseline')} |")
    lines += ["", f"Target held-series OOF TSAUC: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.",
        "", f"Selection status: {summary['adaptive_selection']}.", "", summary["interpretation"],
        "", "These are local development OOF scores, not cloud scores.", "", summary["deployment"], ""]
    text = "\n".join(lines)
    (directory / "RESULTS.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print(text, flush=True)
    return summary
