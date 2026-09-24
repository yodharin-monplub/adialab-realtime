"""Pilot selection and untouched confirmation for Model44."""
import numpy as np
from scipy.special import expit

from models.model10.causal_transforms import apply_causal_transform
from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as incumbent
from models.model35 import run_model35 as anti_residual
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as exploratory
from models.model44.run_model44 import CONFIG, HERE, ROOT, frozen_study, lightgbm_probability
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


def candidate_margins(layout, directory, fingerprint):
    model26_margin, model35_margin, model39_margin = exploratory.margins(layout)
    anti_delta = model35_margin.astype(np.float64) - model26_margin.astype(np.float64)
    base_probability = np.empty(layout.row_count, dtype=np.float32)
    lgb_probability = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        base_probability[held] = incumbent.prediction_checkpoint(None, "depth_mix", fold, None, held)
        lgb_probability[held] = lightgbm_probability(directory, fold, fingerprint, held)
    values = [model39_margin]
    for weight in CONFIG["blend_weights"]:
        probability = ((1 - weight) * base_probability + weight * lgb_probability).astype(np.float32)
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted_probability = expit(replaced + anti_delta)
        margin = apply_causal_transform(adjusted_probability, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32)
        values.append(margin)
    return values


def report():
    directory, _, fingerprint = frozen_study()
    layout = load_development_layout(ROOT)
    values = candidate_margins(layout, directory, fingerprint)
    pilot = np.asarray(CONFIG["pilot_folds"])
    confirmation = np.asarray(CONFIG["confirmation_folds"])
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], value[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for value in values])
    pilot_gains = scores[1:, pilot].mean(axis=1) - scores[0, pilot].mean()
    # Deterministic tie-break toward the smaller, safer blend.
    selected_index = int(np.flatnonzero(pilot_gains == np.max(pilot_gains))[0]) + 1
    selected_weight = CONFIG["blend_weights"][selected_index - 1]
    base = scores[0, confirmation]
    candidate = scores[selected_index, confirmation]
    gain = float(candidate.mean() - base.mean())
    wins = int(np.sum(candidate > base))
    meta = dict(protocol_sha256=fingerprint, selected_weight=selected_weight)
    saved = read_checkpoint(directory / "bootstrap.npz", meta)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in confirmation:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows])
            for value in (values[0], values[selected_index])]))
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
    half = (layout.series_ids * 0x9E3779B1) & 1
    confirm_rows = np.isin(layout.row_fold, confirmation)
    split = []
    for value in (0, 1):
        rows = confirm_rows & (half[layout.series_position] == value)
        split.append(float(tsauc(layout.target[rows], values[selected_index][rows], layout.time_online[rows])
            - tsauc(layout.target[rows], values[0][rows], layout.time_online[rows])))
    reached = bool(candidate.mean() >= CONFIG["target_tsauc"])
    passed = bool(reached and gain >= CONFIG["minimum_confirmation_gain"]
        and wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    result = describe(candidate, base)
    result.update(selected_weight=selected_weight, fold_wins=wins, bootstrap_lower_95=lower,
        split_half_gains=split, passes=passed,
        no_break_false_alarms=false_alarms(expit(values[selected_index].astype(np.float64)), layout, [.5, .8]))
    summary = dict(model=44, selected_weight=selected_weight,
        pilot_weights=CONFIG["blend_weights"], pilot_gains=pilot_gains.tolist(),
        confirmation_model39=describe(base, base), confirmation_candidate=result,
        target_tsauc=CONFIG["target_tsauc"], target_reached=reached,
        selected_candidate="lightgbm_diversity" if passed else None,
        interpretation="Weight selected on folds0-1; folds2-4 untouched confirmation.",
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model44: regularized LightGBM diversity voter", "",
        "| Arm | Confirmation TSAUC | Gain | Wins | Lower 95% | Split gains | Pass |",
        "|---|---:|---:|---:|---:|---|---|",
        f"| model39 | {base.mean():.6f} | +0.000000 | 0/3 | +0.000000 | baseline | baseline |",
        f"| LightGBM weight {selected_weight:.2f} | {candidate.mean():.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {split} | {passed} |",
        "", f"Pilot gains by weight: {dict(zip(CONFIG['blend_weights'], pilot_gains.tolist()))}.",
        f"Confirmation target: {CONFIG['target_tsauc']:.3f}; reached: {reached}.", "",
        summary["interpretation"], "", "These are local held-series OOF scores, not cloud scores.",
        "", summary["deployment"], ""]
    text = "\n".join(lines)
    (directory / "RESULTS.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print(text, flush=True)
    return summary
