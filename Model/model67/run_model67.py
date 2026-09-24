"""Evaluate one frozen 50/25/25 ensemble of Models 62, 63 and 64."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest
from models.model62 import report62, run_model62
from models.model63 import report63, run_model63
from models.model64 import report64, run_model64
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
SOURCES = {"62": (run_model62, report62), "63": (run_model63, report63),
    "64": (run_model64, report64)}


def parent_state():
    parents = {}
    for number, (runner, reporter) in SOURCES.items():
        directory, _, fingerprint = runner.frozen_study()
        pinned = CONFIG["parents"][number]
        if directory != ROOT / pinned["run"] or fingerprint != pinned["protocol_sha256"]:
            raise ValueError(f"Changed Model{number} parent")
        if not (directory / "summary.json").exists():
            raise ValueError(f"Missing Model{number} summary")
        parents[number] = (directory, fingerprint, reporter)
    if set(CONFIG["weights"]) != set(SOURCES) or not np.isclose(
            sum(CONFIG["weights"].values()), 1., rtol=0, atol=1e-12):
        raise ValueError("Invalid frozen ensemble weights")
    return parents


def initialize(parents):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model67.py", HERE / "prespec.json", HERE / "readme.txt",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "train/y_train_index.parquet"]
    for number, (directory, _, _) in parents.items():
        paths.extend((directory / "protocol.json", directory / "summary.json",
            ROOT / f"models/model{number}/report{number}.py",
            ROOT / f"models/model{number}/run_model{number}.py"))
        arm = "lightgbm_aligned_40"
        for fold in range(5):
            paths.append(directory / f"{arm}_fold{fold}.npz")
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths})
    protocol = dict(contract=contract)
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model67 protocol mismatch")
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
        raise ValueError("Changed Model67 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model67 input: {relative}")
    return directory, protocol, fingerprint


def predictions(layout, parents):
    selected = {}
    model39 = None
    for number, (directory, fingerprint, reporter) in parents.items():
        summary = json.loads((directory / "summary.json").read_text())
        values = reporter.candidate_margins(layout, directory, fingerprint, range(5))
        index = summary["pilot"]["selected_index"] + 1
        selected[number] = np.asarray(values[index], dtype=np.float32).copy()
        if number == "62":
            model39 = np.asarray(values[0], dtype=np.float32).copy()
        del values
    if any(value.shape != (layout.row_count,) or not np.isfinite(value).all()
           for value in selected.values()):
        raise ValueError("Invalid parent OOF prediction")
    candidate = sum((CONFIG["weights"][number] * selected[number]
        for number in SOURCES), np.zeros(layout.row_count, dtype=np.float64)).astype(np.float32)
    return selected, candidate, model39


def fold_scores(layout, value):
    return np.asarray([tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)])


def report(directory, fingerprint, layout, selected, candidate, model39):
    base = selected["62"]
    base_scores, scores, m39_scores = (fold_scores(layout, value)
        for value in (base, candidate, model39))
    for number, value in selected.items():
        expected = np.asarray(json.loads((ROOT / CONFIG["parents"][number]["run"] /
            "summary.json").read_text())["candidate_fold_scores"])
        np.testing.assert_allclose(fold_scores(layout, value), expected, rtol=0, atol=1e-12)
    confirm = np.asarray(CONFIG["confirmation_folds"])
    gain = float(scores[confirm].mean() - base_scores[confirm].mean())
    wins = int(np.sum(scores[confirm] > base_scores[confirm]))
    scorers = []
    for fold in confirm:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        pair = [WeightedTSAUC(layout.target[rows], value[rows],
            layout.time_online[rows], layout.series_position[rows])
            for value in (base, candidate)]
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
                dict(protocol_sha256=fingerprint))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows]) -
            tsauc(layout.target[rows], base[rows], layout.time_online[rows])))
    confirmation_score, full_score = float(scores[confirm].mean()), float(scores.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and
        full_score >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_confirmation_gain"] and
        wins >= CONFIG["minimum_confirmation_wins"] and lower > 0 and min(split) > 0)
    summary = dict(model=67, weights=CONFIG["weights"],
        model39_fold_scores=m39_scores.tolist(), baseline_fold_scores=base_scores.tolist(),
        candidate_fold_scores=scores.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model67: fixed 50/25/25 held-series ensemble\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain vs Model62 | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| model39 | {m39_scores[confirm].mean():.6f} | {m39_scores.mean():.6f} | — | — | — | — |\n"
        f"| model62 | {base_scores[confirm].mean():.6f} | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| 50% m62 + 25% m63 + 25% m64 | {confirmation_score:.6f} | {full_score:.6f} | "
        f"{gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Confirmation split-half gains vs Model62: {split}.\n"
        "Exploratory held-series OOF; all three parent model choices reused development folds.\n\n"
        f"{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parents = parent_state()
        directory, _, fingerprint = initialize(parents)
        layout = load_development_layout(ROOT)
        selected, candidate, model39 = predictions(layout, parents)
        report(directory, fingerprint, layout, selected, candidate, model39)


if __name__ == "__main__":
    main()
