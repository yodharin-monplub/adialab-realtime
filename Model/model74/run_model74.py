"""Cross-fitted online-age selector over frozen held-series OOF predictions."""
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
from models.model67 import run_model67 as parent
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def parent_state():
    directory, _, fingerprint = parent.frozen_study()
    if directory != ROOT / CONFIG["parent_run"]:
        raise ValueError("Changed Model67 parent run")
    if fingerprint != CONFIG["parent_protocol_sha256"]:
        raise ValueError("Changed Model67 parent protocol")
    return directory, fingerprint


def initialize(parent_directory):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [
        HERE / "run_model74.py",
        HERE / "prespec.json",
        HERE / "readme.txt",
        ROOT / "models/model67/run_model67.py",
        parent_directory / "protocol.json",
        parent_directory / "summary.json",
        ROOT / "models/model10/sources.py",
        ROOT / "models/report_next_models.py",
        ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "models/artifacts/core_features.metadata.npz",
        ROOT / "train/y_train_index.parquet",
        ROOT / "tsauc.py",
    ]
    protocol = {
        "contract": {
            "config": CONFIG,
            "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        }
    }
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model74 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", {
        "directory": str(directory.relative_to(ROOT)),
        "protocol_sha256": fingerprint,
    })
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"]:
        raise ValueError("Changed Model74 protocol")
    if protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model74 configuration")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model74 input: {relative}")
    return directory, protocol, fingerprint


def sources(layout):
    parents = parent.parent_state()
    selected, ensemble, model39 = parent.predictions(layout, parents)
    values = {
        "m39": np.asarray(model39, dtype=np.float32),
        "m62": np.asarray(selected["62"], dtype=np.float32),
        "m63": np.asarray(selected["63"], dtype=np.float32),
        "m64": np.asarray(selected["64"], dtype=np.float32),
        "m67": np.asarray(ensemble, dtype=np.float32),
    }
    if list(values) != CONFIG["candidate_order"]:
        raise ValueError("Candidate order does not match frozen configuration")
    if any(value.shape != (layout.row_count,) or not np.isfinite(value).all()
           for value in values.values()):
        raise ValueError("Invalid parent OOF prediction")
    return values


def score(rows, layout, value):
    return float(tsauc(layout.target[rows], value[rows], layout.time_online[rows]))


def cross_fitted_prediction(layout, values):
    names = CONFIG["candidate_order"]
    edges = CONFIG["time_edges"]
    baseline = CONFIG["baseline"]
    band_scores = np.empty((len(names), 5, len(edges) - 1), dtype=np.float64)
    for source_index, name in enumerate(names):
        for fold in range(5):
            for band, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                rows = ((layout.row_fold == fold) & (layout.time_online >= lower) &
                        (layout.time_online < upper))
                band_scores[source_index, fold, band] = score(rows, layout, values[name])

    candidate = np.empty(layout.row_count, dtype=np.float32)
    selections = []
    baseline_index = names.index(baseline)
    for held_fold in range(5):
        training_folds = np.asarray([fold for fold in range(5) if fold != held_fold])
        for band, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            training_scores = band_scores[:, training_folds, band].mean(axis=1)
            best_index = int(np.argmax(training_scores))
            gain = float(training_scores[best_index] - training_scores[baseline_index])
            if gain < CONFIG["minimum_training_gain"]:
                best_index = baseline_index
            name = names[best_index]
            rows = ((layout.row_fold == held_fold) & (layout.time_online >= lower) &
                    (layout.time_online < upper))
            candidate[rows] = values[name][rows]
            selections.append({
                "held_fold": held_fold,
                "lower": lower,
                "upper_exclusive": upper,
                "source": name,
                "training_gain": gain,
            })
    if not np.isfinite(candidate).all():
        raise ValueError("Cross-fitted prediction has missing or invalid rows")
    return candidate, band_scores, selections


def fold_scores(layout, value):
    return np.asarray([score(layout.row_fold == fold, layout, value) for fold in range(5)])


def report(directory, fingerprint, layout, values, candidate, band_scores, selections):
    baseline = values[CONFIG["baseline"]]
    base_scores = fold_scores(layout, baseline)
    candidate_scores = fold_scores(layout, candidate)
    fold_gains = candidate_scores - base_scores

    scorers = []
    for fold in range(5):
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
                {"protocol_sha256": fingerprint})
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))

    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = halves[layout.series_position] == half
        split.append(score(rows, layout, candidate) - score(rows, layout, baseline))

    full_score = float(candidate_scores.mean())
    full_gain = float(fold_gains.mean())
    wins = int(np.sum(fold_gains > 0))
    passed = bool(full_score >= CONFIG["target_tsauc"] and
        full_gain >= CONFIG["minimum_full_gain"] and wins >= CONFIG["minimum_fold_wins"] and
        lower > 0 and min(split) > 0)
    summary = {
        "model": 74,
        "baseline": CONFIG["baseline"],
        "baseline_fold_scores": base_scores.tolist(),
        "candidate_fold_scores": candidate_scores.tolist(),
        "fold_gains": fold_gains.tolist(),
        "full_oof_score": full_score,
        "full_oof_gain": full_gain,
        "fold_wins": wins,
        "bootstrap_lower_95": lower,
        "split_half_gains": split,
        "selections": selections,
        "band_scores": band_scores.tolist(),
        "passes": passed,
        "deployment": "No GPU rental, packaging, upload, or cloud run performed.",
    }
    write_json(directory / "summary.json", summary)
    selection_lines = [
        f"- held fold {item['held_fold']}, ages {item['lower']}-{item['upper_exclusive'] - 1}: "
        f"{item['source']} (other-fold gain {item['training_gain']:+.6f})"
        for item in selections if item["source"] != CONFIG["baseline"]
    ]
    if not selection_lines:
        selection_lines = ["- No switch cleared the frozen training-fold gain threshold."]
    result = (
        "# Model74: leave-one-fold-out age-band selector\n\n"
        "| Arm | Full OOF TSAUC | Gain | Fold wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| Cross-fitted selector | {full_score:.6f} | {full_gain:+.6f} | "
        f"{wins}/5 | {lower:+.6f} | {passed} |\n\n"
        f"Fold gains: {fold_gains.tolist()}.\n"
        f"Split-half gains: {split}.\n\n"
        "Switches away from Model67:\n" + "\n".join(selection_lines) + "\n\n"
        "Every held fold's selections were learned from the other four folds only. "
        "Band design and source pool remain development-informed.\n\n"
        f"{summary['deployment']}\n"
    )
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
        values = sources(layout)
        candidate, band_scores, selections = cross_fitted_prediction(layout, values)
        report(directory, fingerprint, layout, values, candidate, band_scores, selections)


if __name__ == "__main__":
    main()
