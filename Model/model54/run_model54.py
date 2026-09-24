"""Test negative blending of the frozen late-feature voter on untouched folds."""
from __future__ import annotations

import argparse
import fcntl
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
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.model53 import run_model53 as late_voter
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT = ROOT / CONFIG["model53_run"]


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    parent_dir, parent, parent_sha = late_voter.frozen_study()
    if parent_dir != PARENT or parent_sha != CONFIG["model53_protocol_sha256"]:
        raise ValueError("Changed Model53 parent")
    d26, _, s26 = voter.frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s26 != CONFIG["model26_protocol_sha256"] or s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed incumbent protocol")
    paths = [HERE / "run_model54.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT / "protocol.json", d26 / "protocol.json", d39 / "protocol.json",
        ROOT / "models/model53/run_model53.py", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model20/report.py", ROOT / "models/model38/run_model38.py",
        ROOT / "models/model10/causal_transforms.py", ROOT / "tsauc.py"]
    for fold in CONFIG["pilot_folds"]:
        paths.extend([PARENT / f"aligned_25_fold{fold}.npz",
            PARENT / f"aligned_25_fold{fold}.json"])
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        recipes=parent["contract"]["recipes"],
        names=parent["contract"]["names"],
        union_names=parent["contract"]["union_names"],
        parent_protocol_sha256=parent_sha)
    protocol = dict(contract=contract, plans=parent["plans"],
        plans_sha256=parent["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model54 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", dict(
        directory=str(directory.relative_to(ROOT)), protocol_sha256=fingerprint))
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model54 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model54 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model54 input: {relative}")
    return directory, protocol, fingerprint


def voter_probabilities(layout, directory, fingerprint, completed_folds):
    d26, _, s26 = voter.frozen_study()
    original = np.empty(layout.row_count, dtype=np.float32)
    candidate = np.empty_like(original)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        original[held] = voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        if fold not in completed_folds:
            candidate[held] = original[held]
            continue
        if fold in CONFIG["pilot_folds"]:
            source, expected = PARENT, CONFIG["model53_protocol_sha256"]
        else:
            source, expected = directory, fingerprint
        saved = read_checkpoint(source / f"aligned_25_fold{fold}.npz",
            dict(protocol_sha256=expected, fold=fold, arm="aligned_25"))
        if saved is None:
            raise ValueError(f"Missing Model54 voter fold {fold}")
        validate_predictions(saved, held)
        candidate[held] = saved["prediction"]
    return original, candidate


def candidate_margins(layout, directory, fingerprint, completed_folds):
    m26, m35, m39 = incumbent.margins(layout)
    anti_delta = m35.astype(np.float64) - m26.astype(np.float64)
    original, candidate = voter_probabilities(layout, directory, fingerprint, completed_folds)
    values, clipped = [m39], [0.]
    for weight in CONFIG["blend_weights"]:
        raw = original + weight * (candidate - original)
        clipped.append(float(np.mean((raw < 1e-6) | (raw > 1 - 1e-6))))
        probability = np.clip(raw, 1e-6, 1 - 1e-6).astype(np.float32)
        replaced = ensemble_margin(probability, layout).astype(np.float64)
        adjusted = expit(replaced + anti_delta)
        values.append(apply_causal_transform(adjusted, layout.series_position,
            smoother.transform_spec(), layout.time_online).astype(np.float32))
    return values, clipped


def fold_scores(layout, values):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for value in values])


def pilot_decision(directory, fingerprint, layout):
    values, clipped = candidate_margins(layout, directory, fingerprint,
        CONFIG["pilot_folds"])
    scores = fold_scores(layout, values)
    pilot = np.asarray(CONFIG["pilot_folds"])
    gains = scores[1:, pilot].mean(axis=1) - scores[0, pilot].mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, pilot_gains=gains.tolist(),
        baseline_scores=scores[0, pilot].tolist(),
        candidate_scores=scores[selected + 1, pilot].tolist(),
        selected_index=selected, selected_weight=CONFIG["blend_weights"][selected],
        clipped_fraction=clipped[selected + 1],
        continue_training=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model54 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, decision):
    values, clipped = candidate_margins(layout, directory, fingerprint, range(5))
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
                dict(protocol_sha256=fingerprint, selected_weight=decision["selected_weight"]))
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
    summary = dict(model=54, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split,
        clipped_fraction=clipped[decision["selected_index"] + 1], passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model54: anti-reselected-voter blend\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| model39 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| anti-voter weight {decision['selected_weight']:.2f} | {confirmation_score:.6f} | "
        f"{full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {decision['pilot_gains']}.\n"
        f"Confirmation split-half gains: {split}; clipped fraction: "
        f"{summary['clipped_fraction']:.6f}.\n"
        "Exploratory held-series OOF; pilot direction and inherited ensemble are development-informed.\n\n"
        f"{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint, layout):
    decision = pilot_decision(directory, fingerprint, layout)
    if not decision["continue_training"]:
        result = ("# Model54: anti-reselected-voter blend\n\n"
            f"Pilot gains: {dict(zip(CONFIG['blend_weights'], decision['pilot_gains']))}.\n"
            "No coefficient met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"aligned_25_fold{fold}.npz").exists()
           for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, decision)
    else:
        print(f"Pilot selected anti-voter weight {decision['selected_weight']:.2f}; "
              "confirmation fits pending", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, protocol, fingerprint = initialize()
        layout = load_development_layout(ROOT)
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        if not report(directory, fingerprint, layout):
            return
        parent, union, _, _, _ = late_voter.parent_state()
        real = late_voter.real_cache(layout, union)
        pool = load_real_pool(layout)
        for fold in CONFIG["confirmation_folds"]:
            late_voter.run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
        report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
