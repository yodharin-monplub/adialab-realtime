"""Unseen confirmation report for the Model29 reliability gate."""
import json
import numpy as np

from models.model11.pipeline import describe, sha256, write_json
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model20.report import ensemble_margin
from models.model28.report import false_alarms
from models.model29.run_model29 import (ROOT, HERE, ARMS, CONFIG, frozen_study,
    raw_predictions, fit_gate, gated_probability)
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


def report(layout, context):
    directory, _, fingerprint = frozen_study()
    gate = fit_gate(directory, fingerprint, layout, context)
    component = [np.empty(layout.row_count, dtype=np.float32) for _ in ARMS]
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        base, new = raw_predictions(fold, held)
        component[0][held] = base
        component[1][held] = gated_probability(gate, base, new,
            context[layout.series_position[held]], layout.time_online[held])
    margins = [ensemble_margin(p, layout) for p in component]
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold], margin[layout.row_fold == fold],
        layout.time_online[layout.row_fold == fold]) for fold in range(5)] for margin in margins])
    confirm = np.asarray(CONFIG["confirmation_folds"])
    gain = float(scores[1, confirm].mean() - scores[0, confirm].mean())
    wins = int(np.sum(scores[1, confirm] > scores[0, confirm]))
    half = (layout.series_ids * 0x9E3779B1) & 1
    split_gains = []
    for value in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (half[layout.series_position] == value)
        split_gains.append(float(tsauc(layout.target[rows], margins[1][rows], layout.time_online[rows])
            - tsauc(layout.target[rows], margins[0][rows], layout.time_online[rows])))
    model27_dir = ROOT / json.loads((ROOT / "models/model27/runs/latest.json").read_text())["directory"]
    hashes = {f"fold{fold}": sha256(model27_dir / f"matched_context_fold{fold}.npz") for fold in range(5)}
    meta = dict(protocol_sha256=fingerprint, predictions=hashes, gate=sha256(directory / "gate.npz"))
    saved = read_checkpoint(directory / "bootstrap.npz", meta)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in confirm:
        rows = layout.row_fold == fold; series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], margin[rows],
            layout.time_online[rows], layout.series_position[rows]) for margin in margins]))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"]); completed = len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local = []
        for series, pair in scorers:
            multiplicity = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            if repeat < completed: continue
            weights = np.zeros(len(layout.inner_series_fold)); weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in pair])
        if repeat < completed: continue
        local = np.asarray(local).mean(axis=0); boot.append(float(local[1] - local[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)), meta)
    lower = float(np.quantile(boot, (1-CONFIG["confidence"])/2))
    confirmation_score = float(scores[1, confirm].mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_confirmation_gain"]
        and wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split_gains) > 0)
    results = {arm: dict(five_fold=describe(scores[i], scores[0]),
        pilot_scores=scores[i, CONFIG["pilot_folds"]].tolist(),
        confirmation_scores=scores[i, confirm].tolist(), confirmation_mean=float(scores[i, confirm].mean()))
        for i, arm in enumerate(ARMS)}
    results["context_gate"].update(confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_confirmation_gains=split_gains, passes=passed)
    summary = dict(model=29, selected_candidate="context_gate" if passed else None,
        target_tsauc=CONFIG["target_tsauc"], target_reached=bool(confirmation_score >= CONFIG["target_tsauc"]),
        results=results, interpretation=CONFIG["interpretation"],
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model29: pilot-trained context reliability gate", "",
        "| Arm | Confirmation folds 2-4 | Gain | Wins | Lower 95% | Split gains | Pass |",
        "|---|---:|---:|---:|---:|---|---|"]
    for arm in ARMS:
        result = results[arm]
        lines.append(f"| {arm} | {result['confirmation_mean']:.6f} | {result.get('confirmation_gain',0):+.6f} | "
            f"{result.get('confirmation_wins',0)}/3 | {result.get('bootstrap_lower_95',0):+.6f} | "
            f"{result.get('split_half_confirmation_gains','baseline')} | {result.get('passes','baseline')} |")
    lines += ["", f"Unseen confirmation target: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.",
        "Folds 0-1 train the gate and are excluded from confirmation.", "", CONFIG["interpretation"], "",
        "These are local held-series OOF scores, not cloud scores.", "", summary["deployment"], ""]
    rendered = "\n".join(lines); (directory / "RESULTS.md").write_text(rendered); (HERE / "RESULTS.md").write_text(rendered)
    print(rendered, flush=True); return summary
