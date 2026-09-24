"""Label-free absorbing-break evidence filter with pilot/confirmation gate."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest
from models.model39 import run_model39 as incumbent
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def evidence_series(margin, hazard, scale, center):
    """Causal excess log odds from an absorbing-state heuristic."""
    m = np.asarray(margin, dtype=np.float64)
    if m.ndim != 1 or not np.isfinite(m).all() or not 0 < hazard < 1:
        raise ValueError("Invalid Model56 series or hazard")
    output = np.empty(len(m), dtype=np.float32)
    q = 0.
    log_survival = math.log1p(-hazard)
    for step, value in enumerate(m):
        prior = min(max(q + (1 - q) * hazard, 1e-10), 1 - 1e-10)
        posterior_logit = math.log(prior / (1 - prior)) + scale * (value - center)
        q = 1 / (1 + math.exp(-max(min(posterior_logit, 40), -40)))
        q = min(max(q, 1e-10), 1 - 1e-10)
        neutral_prior = min(max(-math.expm1((step + 1) * log_survival), 1e-10), 1 - 1e-10)
        neutral_logit = math.log(neutral_prior / (1 - neutral_prior))
        output[step] = np.float32((math.log(q / (1 - q)) - neutral_logit) /
                                      math.sqrt(step + 1))
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite Model56 evidence")
    return output


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed Model39 protocol")
    paths = [HERE / "run_model56.py", HERE / "prespec.json", HERE / "readme.txt",
        d39 / "protocol.json", ROOT / "models/model39/run_model39.py",
        ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
        ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "train/y_train_index.parquet"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths}))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model56 protocol mismatch")
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
        raise ValueError("Changed Model56 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model56 input: {relative}")
    return directory, protocol, fingerprint


def evidence_cache(directory, fingerprint, layout, margin):
    path = directory / "excess_evidence.npy"
    manifest = path.with_suffix(".json")
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record != dict(protocol_sha256=fingerprint, rows=layout.row_count,
                          sha256=sha256(path)):
            raise ValueError("Corrupt Model56 evidence cache")
        result = np.load(path, mmap_mode="r", allow_pickle=False)
        if result.shape != (layout.row_count,) or result.dtype != np.float32:
            raise ValueError("Wrong Model56 evidence-cache shape")
        return result
    positions = layout.series_position
    output = np.empty(layout.row_count, dtype=np.float32)
    for number, position in enumerate(np.flatnonzero(layout.inner_series_fold >= 0)):
        left, right = np.searchsorted(positions, (position, position + 1))
        series = margin[left:right]
        result = evidence_series(series, CONFIG["hazard"],
            CONFIG["evidence_scale"], CONFIG["evidence_center"])
        if number < 3:
            for cut in (1, 16, 64, 256):
                if cut <= len(series):
                    np.testing.assert_array_equal(result[cut - 1],
                        evidence_series(series[:cut], CONFIG["hazard"],
                            CONFIG["evidence_scale"], CONFIG["evidence_center"])[-1])
        output[left:right] = result
    if len(output) != layout.row_count or not np.isfinite(output).all():
        raise ValueError("Incomplete Model56 evidence")
    temporary = directory / "excess_evidence.partial.npy"
    with temporary.open("wb") as file:
        np.save(file, output, allow_pickle=False)
    temporary.replace(path)
    write_json(manifest, dict(protocol_sha256=fingerprint,
        rows=layout.row_count, sha256=sha256(path)))
    return output


def score_folds(layout, values, folds):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in folds] for value in values])


def pilot_decision(directory, fingerprint, layout, values):
    scores = score_folds(layout, values, CONFIG["pilot_folds"])
    gains = scores[1:].mean(axis=1) - scores[0].mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, pilot_gains=gains.tolist(),
        baseline_scores=scores[0].tolist(),
        candidate_scores=scores[selected + 1].tolist(),
        selected_index=selected, selected_weight=CONFIG["blend_weights"][selected],
        continue_evaluation=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model56 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, values, decision):
    scores = score_folds(layout, values, range(5))
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
    summary = dict(model=56, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model56: absorbing-break evidence filter\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| model39 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| filter weight {decision['selected_weight']:.2f} | {confirmation_score:.6f} | "
        f"{full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {decision['pilot_gains']}.\n"
        f"Confirmation split-half gains: {split}.\n"
        "Exploratory held-series OOF; Model39 and filter direction are development-informed.\n\n"
        f"{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, _, fingerprint = initialize()
        layout = load_development_layout(ROOT)
        margin = incumbent.margins(layout)[2]
        evidence = evidence_cache(directory, fingerprint, layout, margin)
        values = [margin] + [(margin + weight * evidence).astype(np.float32)
            for weight in CONFIG["blend_weights"]]
        decision = pilot_decision(directory, fingerprint, layout, values)
        if decision["continue_evaluation"]:
            final_report(directory, fingerprint, layout, values, decision)
        else:
            result = ("# Model56: absorbing-break evidence filter\n\n"
                f"Pilot gains: {dict(zip(CONFIG['blend_weights'], decision['pilot_gains']))}.\n"
                "No weight met the prespecified +0.0001 pilot gate; confirmation folds were not inspected.\n"
                "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
            (directory / "RESULTS.md").write_text(result)
            (HERE / "RESULTS.md").write_text(result)
            print(result, flush=True)


if __name__ == "__main__":
    main()
