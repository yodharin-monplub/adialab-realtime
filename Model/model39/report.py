"""Whole-series evaluation of Model39 against Models26 and 35."""
import numpy as np
from scipy.special import expit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model39.run_model39 import CONFIG, HERE, ROOT, frozen_study, margins
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


EXPECTED_MODEL26 = np.asarray([.6427355043536926, .6085023046201091, .6422531072481089,
                               .6218047904181766, .6467391865285458])


def report():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    values = margins(layout)
    names = ("model26", "model35", "model39")
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], margin[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for margin in values])
    np.testing.assert_allclose(scores[0], EXPECTED_MODEL26, rtol=0, atol=1e-12)
    split = []
    half = (layout.series_ids * 0x9E3779B1) & 1
    for value in (0, 1):
        rows = half[layout.series_position] == value
        split.append([float(tsauc(layout.target[rows], values[index][rows], layout.time_online[rows])
                    - tsauc(layout.target[rows], values[0][rows], layout.time_online[rows]))
                    for index in (1, 2)])
    metadata = dict(protocol_sha256=fingerprint,
        source_protocols=[CONFIG["model35_protocol_sha256"], CONFIG["model38_protocol_sha256"]])
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
        for series, triple in scorers:
            multiplicity = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            if repeat < completed:
                continue
            weights = np.zeros(len(layout.inner_series_fold))
            weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in triple])
        if repeat < completed:
            continue
        mean = np.asarray(local).mean(axis=0)
        boot.append([float(mean[2] - mean[0]), float(mean[2] - mean[1])])
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)), metadata)
    boot = np.asarray(boot)
    lower = np.quantile(boot, (1 - CONFIG["confidence"]) / 2, axis=0)
    gain26 = float(scores[2].mean() - scores[0].mean())
    gain35 = float(scores[2].mean() - scores[1].mean())
    wins26 = int(np.sum(scores[2] > scores[0]))
    wins35 = int(np.sum(scores[2] > scores[1]))
    split39 = [row[1] for row in split]
    passed = bool(scores[2].mean() >= CONFIG["target_tsauc"]
        and gain26 >= CONFIG["minimum_gain_vs_model26"]
        and wins26 >= CONFIG["minimum_fold_wins_vs_model26"]
        and wins35 >= CONFIG["minimum_fold_wins_vs_model35"]
        and np.min(lower) > 0 and min(split39) > 0)
    results = {name: describe(score, scores[0]) for name, score in zip(names, scores)}
    results["model39"].update(gain_vs_model35=gain35, fold_wins_vs_model26=wins26,
        fold_wins_vs_model35=wins35, bootstrap_lower_95_vs_model26=float(lower[0]),
        bootstrap_lower_95_vs_model35=float(lower[1]), split_half_gains_vs_model26=split39,
        passes=passed, no_break_false_alarms=false_alarms(expit(values[2].astype(np.float64)), layout, [.5, .8]))
    summary = dict(model=39, target_tsauc=CONFIG["target_tsauc"],
        target_reached=bool(scores[2].mean() >= CONFIG["target_tsauc"]),
        selected_candidate="model39" if passed else None, results=results,
        interpretation="Exploratory Model35 margin with independently selected and confirmed Model38 smoother.",
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model39: smoothed anti-residual composition", "",
        "| Arm | Held-series OOF | Gain vs M26 | Wins vs M26 | Lower vs M26 | Pass |",
        "|---|---:|---:|---:|---:|---|",
        f"| model26 | {scores[0].mean():.6f} | +0.000000 | 0/5 | +0.000000 | baseline |",
        f"| model35 | {scores[1].mean():.6f} | {scores[1].mean()-scores[0].mean():+.6f} | {int(np.sum(scores[1]>scores[0]))}/5 | diagnostic | exploratory |",
        f"| model39 | {scores[2].mean():.6f} | {gain26:+.6f} | {wins26}/5 | {lower[0]:+.6f} | {passed} |",
        "", f"Model39 gain vs Model35: {gain35:+.6f}; wins {wins35}/5; bootstrap lower 95% {lower[1]:+.6f}.",
        f"Split-half gains vs Model26: {split39}.",
        f"Target: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.", "",
        summary["interpretation"], "", "These are local development OOF scores, not cloud scores.",
        "", summary["deployment"], ""]
    text = "\n".join(lines)
    (directory / "RESULTS.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print(text, flush=True)
    return summary

