"""Select a causal persistence transform on two folds and confirm on three."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.causal_transforms import TransformSpec, apply_transform_grid
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest
from models.model67 import run_model67 as parent
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def parent_state():
    directory, _, fingerprint = parent.frozen_study()
    if directory != ROOT / CONFIG["parent_run"] or fingerprint != CONFIG["parent_protocol_sha256"]:
        raise ValueError("Changed Model83 parent")
    return directory, fingerprint


def initialize(parent_directory):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model83.py", HERE / "prespec.json", HERE / "readme.txt",
        ROOT / "models/model67/run_model67.py", parent_directory / "protocol.json",
        parent_directory / "summary.json", ROOT / "models/model10/causal_transforms.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "train/y_train_index.parquet", ROOT / "tsauc.py"]
    protocol = {"contract": {"config": CONFIG,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths}}}
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model83 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", {"directory": str(directory.relative_to(ROOT)),
        "protocol_sha256": fingerprint})
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model83 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model83 input: {relative}")
    return directory, protocol, fingerprint


def specs():
    result = []
    for item in CONFIG["candidates"]:
        values = dict(correction_weight=0.0, half_life=None, window=None,
            decay_factor=None, minimum_scale=None, innovation_clip=None)
        values.update(item)
        spec = TransformSpec(**values)
        spec.validate()
        result.append(spec)
    if result[0].kind != "raw" or len({item.name for item in result}) != len(result):
        raise ValueError("Invalid Model83 transform grid")
    return result


def fold_scores(layout, value):
    return np.asarray([tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)])


def predictions(layout):
    parents = parent.parent_state()
    _, baseline, _ = parent.predictions(layout, parents)
    probability = expit(np.asarray(baseline, dtype=np.float64))
    candidates, score_rows = [], []
    for number, (spec, value) in enumerate(apply_transform_grid(probability,
            layout.time_online, layout.series_position, specs())):
        values = np.asarray(value, dtype=np.float32).copy()
        candidates.append(values)
        scores = fold_scores(layout, values)
        score_rows.append(scores)
        print(f"Model83 candidate {number + 1}/{len(CONFIG['candidates'])} "
              f"{spec.name}: {scores.tolist()}", flush=True)
    return np.asarray(baseline, dtype=np.float32), candidates, np.asarray(score_rows)


def report(directory, fingerprint, layout, baseline, candidates, scores):
    pilot = np.asarray(CONFIG["pilot_folds"])
    confirmation = np.asarray(CONFIG["confirmation_folds"])
    baseline_scores = fold_scores(layout, baseline)
    np.testing.assert_allclose(scores[0], baseline_scores, rtol=0, atol=2e-7)
    pilot_gains = scores[:, pilot].mean(axis=1) - baseline_scores[pilot].mean()
    selected = int(np.argmax(pilot_gains))
    if pilot_gains[selected] < CONFIG["pilot_minimum_gain"]:
        selected = 0
    candidate = candidates[selected]
    candidate_scores = scores[selected]
    gains = candidate_scores - baseline_scores
    scorers = []
    for fold in confirmation:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        pair = [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows])
            for value in (baseline, candidate)]
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
            commit_checkpoint(directory / "bootstrap.npz", {"gains": np.asarray(boot)},
                {"protocol_sha256": fingerprint, "selected_index": selected})
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirmation) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows]) -
            tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
    confirmation_gain = float(gains[confirmation].mean())
    confirmation_score = float(candidate_scores[confirmation].mean())
    full_score = float(candidate_scores.mean())
    wins = int(np.sum(gains[confirmation] > 0))
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and
        full_score >= CONFIG["target_tsauc"] and
        confirmation_gain >= CONFIG["minimum_confirmation_gain"] and
        wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    names = [item.name for item in specs()]
    summary = {"model": 83, "selected_index": selected, "selected_transform": names[selected],
        "pilot_gains": dict(zip(names, pilot_gains.tolist())),
        "baseline_fold_scores": baseline_scores.tolist(),
        "candidate_fold_scores": candidate_scores.tolist(), "fold_gains": gains.tolist(),
        "confirmation_score": confirmation_score, "full_oof_score": full_score,
        "confirmation_gain": confirmation_gain, "confirmation_wins": wins,
        "bootstrap_lower_95": lower, "split_half_gains": split, "passes": passed,
        "deployment": "No packaging, upload, or cloud run performed."}
    write_json(directory / "summary.json", summary)
    result = ("# Model83: causal persistence over Model67\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| Model67 | {baseline_scores[confirmation].mean():.6f} | {baseline_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| {names[selected]} | {confirmation_score:.6f} | {full_score:.6f} | "
        f"{confirmation_gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gain: {pilot_gains[selected]:+.6f}. Fold gains: {gains.tolist()}.\n"
        f"Confirmation split-half gains: {split}.\n\n{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_directory, _ = parent_state()
        directory, _, fingerprint = initialize(parent_directory)
        layout = load_development_layout(ROOT)
        baseline, candidates, scores = predictions(layout)
        report(directory, fingerprint, layout, baseline, candidates, scores)


if __name__ == "__main__":
    main()
