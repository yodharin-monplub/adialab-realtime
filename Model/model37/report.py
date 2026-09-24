"""Paired held-series evaluation for the Model37 seed bag."""
import numpy as np
from scipy.special import expit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as incumbent
from models.model37.run_model37 import CONFIG, HERE, ROOT, frozen_study, replica_probability
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


EXPECTED_MODEL26 = np.asarray([.6427355043536926, .6085023046201091, .6422531072481089,
                               .6218047904181766, .6467391865285458])


def bag_probability(original, replica, weight=0.5):
    original = np.asarray(original, dtype=np.float32)
    replica = np.asarray(replica, dtype=np.float32)
    if original.shape != replica.shape or original.ndim != 1:
        raise ValueError("Bag inputs must be matching vectors")
    if not np.isfinite(original).all() or not np.isfinite(replica).all():
        raise ValueError("Nonfinite bag input")
    return ((1 - weight) * original + weight * replica).astype(np.float32)


def report():
    directory, _, fingerprint = frozen_study()
    incumbent_dir, _, incumbent_sha = incumbent.frozen_study()
    layout = load_development_layout(ROOT)
    original = np.empty(layout.row_count, dtype=np.float32)
    replica = np.empty_like(original)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        original[held] = incumbent.prediction_checkpoint(
            incumbent_dir, "depth_mix", fold, incumbent_sha, held)
        replica[held] = replica_probability(directory, fold, fingerprint, held)
    bag = bag_probability(original, replica, CONFIG["bag_weight"])
    margins = [ensemble_margin(original, layout), ensemble_margin(replica, layout),
               ensemble_margin(bag, layout)]
    names = ("model26", "replica", "seed_bag")
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], value[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for value in margins])
    np.testing.assert_allclose(scores[0], EXPECTED_MODEL26, rtol=0, atol=1e-12)
    half = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for value in (0, 1):
        rows = half[layout.series_position] == value
        split.append(float(tsauc(layout.target[rows], margins[2][rows], layout.time_online[rows])
                           - tsauc(layout.target[rows], margins[0][rows], layout.time_online[rows])))
    prediction_hashes = {path.name: sha256(path) for path in sorted(directory.glob("replica_*.npz"))}
    metadata = dict(protocol_sha256=fingerprint, predictions=prediction_hashes)
    saved = read_checkpoint(directory / "bootstrap.npz", metadata)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in range(5):
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], margin[rows],
            layout.time_online[rows], layout.series_position[rows]) for margin in (margins[0], margins[2])]))
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
    gain = float(scores[2].mean() - scores[0].mean())
    wins = int(np.sum(scores[2] > scores[0]))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    passed = bool(scores[2].mean() >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_gain"]
                  and wins >= CONFIG["minimum_fold_wins"] and lower > 0 and min(split) > 0)
    results = {name: describe(score, scores[0]) for name, score in zip(names, scores)}
    results["seed_bag"].update(fold_wins=wins, bootstrap_lower_95=lower,
        split_half_gains=split, passes=passed,
        no_break_false_alarms=false_alarms(expit(margins[2].astype(np.float64)), layout, [.5, .8]))
    summary = dict(model=37, selected_candidate="seed_bag" if passed else None,
        target_tsauc=CONFIG["target_tsauc"], target_reached=bool(scores[2].mean() >= CONFIG["target_tsauc"]),
        results=results,
        interpretation="Independent XGBoost seed bag; all data, features, folds, weights and hyperparameters fixed.",
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model37: independent-seed depth-voter bag", "",
        "| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |",
        "|---|---:|---:|---:|---:|---|---|"]
    for name in names:
        result = results[name]
        lines.append(f"| {name} | {result['fold_mean']:.6f} | {result['mean_gain']:+.6f} | "
            f"{result.get('fold_wins', 0)}/5 | {result.get('bootstrap_lower_95', 0):+.6f} | "
            f"{result.get('split_half_gains', 'baseline')} | {result.get('passes', 'diagnostic')} |")
    lines += ["", f"Target held-series OOF TSAUC: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.",
        "", summary["interpretation"], "", "These are local development OOF scores, not cloud scores.",
        "", summary["deployment"], ""]
    text = "\n".join(lines)
    (directory / "RESULTS.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print(text, flush=True)
    return summary

