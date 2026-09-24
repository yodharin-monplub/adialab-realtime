"""Cross-fit a regularized pairwise simplex over diverse frozen OOF sources."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint
from models.model45 import run_model45
from models.model48 import run_model48
from models.model57 import report57, run_model57
from models.model59 import report59, run_model59
from models.model67 import run_model67
from models.model76.run_model76 import within_time_rank
from models.model81 import run_model81
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def validate_parent(name, directory, fingerprint):
    expected_run, expected_sha = CONFIG["parent_runs"][name]
    if str(directory.relative_to(ROOT)) != expected_run or fingerprint != expected_sha:
        raise ValueError(f"Changed Model82 parent: {name}")


def parent_state():
    d67, _, s67 = run_model67.frozen_study()
    d57, _, s57 = run_model57.frozen_study()
    d59, _, s59 = run_model59.frozen_study()
    d45, _, s45 = run_model45.frozen_study()
    d48, _, s48 = run_model48.frozen_study()
    d81, _, s81 = run_model81.frozen_study()
    states = {"m67": (d67, s67), "m57": (d57, s57), "m59": (d59, s59),
              "kernel45": (d45, s45), "rank48": (d48, s48), "rank81": (d81, s81)}
    for name, state in states.items():
        validate_parent(name, *state)
    return states


def initialize(states):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model82.py", HERE / "prespec.json", HERE / "readme.txt",
        ROOT / "models/model45/run_model45.py", ROOT / "models/model48/run_model48.py",
        ROOT / "models/model57/run_model57.py", ROOT / "models/model57/report57.py",
        ROOT / "models/model59/run_model59.py", ROOT / "models/model59/report59.py",
        ROOT / "models/model67/run_model67.py", ROOT / "models/model76/run_model76.py",
        ROOT / "models/model81/run_model81.py", ROOT / "models/model10/sources.py",
        ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
        ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    for directory, _ in states.values():
        paths.append(directory / "protocol.json")
        if (directory / "summary.json").exists():
            paths.append(directory / "summary.json")
    for fold in range(5):
        paths.extend((states["rank48"][0] / f"rank_fold{fold}.json",
                      states["rank81"][0] / f"whitened_time_rank_fold{fold}.json"))
    paths.append(states["kernel45"][0] / "kernel_signal.npy")
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths}))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model82 protocol mismatch")
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
        raise ValueError("Changed Model82 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model82 input: {relative}")
    return directory, protocol, fingerprint


def selected_report_prediction(layout, runner, reporter):
    directory, _, fingerprint = runner.frozen_study()
    summary = json.loads((directory / "summary.json").read_text())
    values = reporter.candidate_margins(layout, directory, fingerprint, range(5))
    selected = np.asarray(values[summary["pilot"]["selected_index"] + 1], dtype=np.float32).copy()
    del values
    return selected


def signed_checkpoint(directory, fingerprint, layout, arm, filename):
    values = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        saved = read_checkpoint(directory / filename.format(fold=fold),
            dict(protocol_sha256=fingerprint, fold=fold, arm=arm))
        if (saved is None or not np.array_equal(saved["held_rows"], held)
                or saved["prediction"].shape != held.shape
                or not np.isfinite(saved["prediction"]).all()):
            raise ValueError(f"Invalid signed {arm} fold {fold}")
        values[held] = saved["prediction"]
    return values


def source_values(layout, states):
    parents = run_model67.parent_state()
    selected, m67, m39 = run_model67.predictions(layout, parents)
    values = {
        "m67": np.asarray(m67, dtype=np.float32),
        "m39": np.asarray(m39, dtype=np.float32),
        "m57": selected_report_prediction(layout, run_model57, report57),
        "m59": selected_report_prediction(layout, run_model59, report59),
        "m62": np.asarray(selected["62"], dtype=np.float32),
        "m63": np.asarray(selected["63"], dtype=np.float32),
        "m64": np.asarray(selected["64"], dtype=np.float32),
        "kernel45": np.asarray(np.load(states["kernel45"][0] / "kernel_signal.npy",
            mmap_mode="r", allow_pickle=False), dtype=np.float32),
        "rank48": signed_checkpoint(states["rank48"][0], states["rank48"][1], layout,
            "time_rank", "rank_fold{fold}.npz"),
        "rank81": signed_checkpoint(states["rank81"][0], states["rank81"][1], layout,
            "whitened_time_rank", "whitened_time_rank_fold{fold}.npz"),
    }
    if list(values) != CONFIG["sources"]:
        raise ValueError("Model82 source order changed")
    if any(value.shape != (layout.row_count,) or not np.isfinite(value).all()
           for value in values.values()):
        raise ValueError("Invalid Model82 source prediction")
    return values


def rank_matrix(layout, values):
    matrix = np.empty((layout.row_count, len(CONFIG["sources"])), dtype=np.float32)
    for column, name in enumerate(CONFIG["sources"]):
        for fold in range(5):
            rows = layout.row_fold == fold
            matrix[rows, column] = within_time_rank(values[name][rows], layout.time_online[rows])
        print(f"Model82 ranked source {column + 1}/{len(CONFIG['sources'])}: {name}", flush=True)
    return matrix


def sample_fold_pairs(layout, ranks, fold):
    rows = np.flatnonzero(layout.row_fold == fold)
    target = layout.target[rows]
    online = layout.time_online[rows]
    maximum = int(online.max(initial=0)) + 1
    counts = np.bincount(online, minlength=maximum)
    positives = np.bincount(online, weights=target, minlength=maximum)
    pair_mass = positives * (counts - positives)
    useful = np.flatnonzero(pair_mass > 0)
    rng = np.random.default_rng(CONFIG["pair_seed"] + fold)
    sampled_step = rng.choice(useful, CONFIG["pair_samples_per_fold"], replace=True,
                              p=pair_mass[useful] / pair_mass[useful].sum())
    positive_rows = np.empty(len(sampled_step), dtype=np.int64)
    negative_rows = np.empty(len(sampled_step), dtype=np.int64)
    for step in np.unique(sampled_step):
        locations = np.flatnonzero(sampled_step == step)
        local = rows[online == step]
        local_y = layout.target[local]
        positive_rows[locations] = rng.choice(local[local_y == 1], len(locations), replace=True)
        negative_rows[locations] = rng.choice(local[local_y == 0], len(locations), replace=True)
    difference = ranks[positive_rows] - ranks[negative_rows]
    if difference.shape != (CONFIG["pair_samples_per_fold"], len(CONFIG["sources"])):
        raise ValueError("Wrong Model82 sampled-pair matrix")
    return difference.astype(np.float64)


def fit_weights(difference):
    anchor = np.zeros(len(CONFIG["sources"]), dtype=np.float64)
    anchor[CONFIG["sources"].index(CONFIG["anchor"])] = 1.0
    penalty = float(CONFIG["l2_anchor"])

    def objective(weight):
        margin = difference @ weight
        loss = float(np.logaddexp(0.0, -margin).mean() +
                     penalty * np.sum((weight - anchor) ** 2))
        gradient = (-(difference.T @ expit(-margin)) / len(margin) +
                    2 * penalty * (weight - anchor))
        return loss, gradient

    result = minimize(objective, anchor, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * len(anchor),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                     "jac": lambda w: np.ones_like(w)},
        options={"maxiter": CONFIG["optimizer_maxiter"], "ftol": 1e-12, "disp": False})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Model82 optimizer failed: {result.message}")
    weight = np.maximum(result.x, 0)
    weight /= weight.sum()
    return weight, dict(success=bool(result.success), iterations=int(result.nit),
                        objective=float(result.fun), message=str(result.message))


def cross_fitted(layout, ranks, differences):
    candidate = np.empty(layout.row_count, dtype=np.float32)
    records = []
    for held in range(5):
        training = np.concatenate([differences[fold] for fold in range(5) if fold != held])
        weight, optimizer = fit_weights(training)
        rows = layout.row_fold == held
        candidate[rows] = (ranks[rows] @ weight).astype(np.float32)
        records.append(dict(held_fold=held, weights=dict(zip(CONFIG["sources"], weight.tolist())),
                            optimizer=optimizer))
        active = {name: value for name, value in zip(CONFIG["sources"], weight)
                  if value > 1e-5}
        print(f"Model82 held fold {held} weights: {active}", flush=True)
    if not np.isfinite(candidate).all():
        raise ValueError("Invalid cross-fitted Model82 prediction")
    return candidate, records


def fold_scores(layout, value):
    return np.asarray([tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)])


def report(directory, fingerprint, layout, baseline, candidate, records):
    base_scores = fold_scores(layout, baseline)
    candidate_scores = fold_scores(layout, candidate)
    gains = candidate_scores - base_scores
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
            commit_checkpoint(directory / "bootstrap.npz", dict(gains=np.asarray(boot)),
                dict(protocol_sha256=fingerprint))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = halves[layout.series_position] == half
        split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows]) -
                           tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
    full_score = float(candidate_scores.mean())
    gain = float(gains.mean())
    wins = int(np.sum(gains > 0))
    passed = bool(full_score >= CONFIG["target_tsauc"] and gain >= CONFIG["minimum_full_gain"]
        and wins >= CONFIG["minimum_fold_wins"] and lower > 0 and min(split) > 0)
    summary = dict(model=82, baseline_fold_scores=base_scores.tolist(),
        candidate_fold_scores=candidate_scores.tolist(), fold_gains=gains.tolist(),
        full_oof_score=full_score, full_oof_gain=gain, fold_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, coefficients=records,
        passes=passed, deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model82: cross-fitted pairwise simplex stack\n\n"
        "| Arm | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| Cross-fitted stack | {full_score:.6f} | {gain:+.6f} | {wins}/5 | {lower:+.6f} | {passed} |\n\n"
        f"Fold gains: {gains.tolist()}.\n"
        f"Split-half gains: {split}.\n\n"
        "Every held fold's coefficients were learned from the other four folds only.\n\n"
        f"{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = parent_state()
        directory, _, fingerprint = initialize(states)
        layout = load_development_layout(ROOT)
        values = source_values(layout, states)
        baseline = values["m67"].copy()
        ranks = rank_matrix(layout, values)
        del values
        differences = [sample_fold_pairs(layout, ranks, fold) for fold in range(5)]
        candidate, records = cross_fitted(layout, ranks, differences)
        report(directory, fingerprint, layout, baseline, candidate, records)


if __name__ == "__main__":
    main()
