"""Nested cross-fit a time-specific empirical-null scale-GLR correction."""
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
from models.model91.run_model91 import fold_scores, probabilities
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT84 = ROOT / CONFIG["model84_run"]
PARENT95 = ROOT / CONFIG["model95_run"]


def initialize():
    if (HERE / "runs/latest.json").exists():
        latest = json.loads((HERE / "runs/latest.json").read_text())
        directory = ROOT / latest["directory"]
        protocol = json.loads((directory / "protocol.json").read_text())
        fingerprint = sha256(directory / "protocol.json")
        if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
            raise ValueError("Changed Model104 protocol")
        for relative, expected in protocol["contract"]["hashes"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Changed frozen Model104 input: {relative}")
        return directory, fingerprint
    if (sha256(PARENT84 / "protocol.json") != CONFIG["model84_protocol_sha256"] or
            sha256(PARENT95 / "protocol.json") != CONFIG["model95_protocol_sha256"]):
        raise ValueError("Changed Model104 parent")
    paths = [HERE / "run_model104.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT84 / "protocol.json", PARENT84 / "summary.json",
        PARENT95 / "protocol.json", PARENT95 / "pilot_decision.json",
        ROOT / "models/model95/real_glr.npy", ROOT / "models/model95/glr_audit.json",
        ROOT / "models/model91/run_model91.py", ROOT / "models/model89/bayes_evidence.py",
        ROOT / "models/report_next_models.py", ROOT / "models/model10/sources.py",
        ROOT / "models/model5/artifacts/split_assignments.npz",
        ROOT / "train/y_train_index.parquet", ROOT / "tsauc.py"]
    protocol = {"contract": {"config": CONFIG,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths}}}
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", protocol)
    fingerprint = sha256(directory / "protocol.json")
    write_json(HERE / "runs/latest.json", {
        "directory": str(directory.relative_to(ROOT)), "protocol_sha256": fingerprint})
    return directory, fingerprint


def null_tail_feature(layout, evidence, reference_folds, apply_rows, rows_by_time):
    """Transform apply rows using null samples from disjoint reference folds."""
    apply_rows = np.asarray(apply_rows, dtype=np.int64)
    result = np.empty(len(apply_rows), dtype=np.float32)
    apply_time = layout.time_online[apply_rows]
    apply_order = np.argsort(apply_time, kind="stable")
    sorted_apply_time = apply_time[apply_order]
    for step in np.unique(sorted_apply_time):
        lo, hi = np.searchsorted(sorted_apply_time, (step, step + 1))
        local = apply_order[lo:hi]
        time_rows = rows_by_time[int(step)]
        reference_rows = time_rows[(np.isin(layout.row_fold[time_rows], reference_folds)) &
                                   (layout.target[time_rows] == 0)]
        reference = np.sort(np.asarray(evidence[reference_rows], dtype=np.float64))
        if len(reference) < 100:
            raise ValueError(f"Too few Model104 null references at step {step}: {len(reference)}")
        observed = np.asarray(evidence[apply_rows[local]], dtype=np.float64)
        left = np.searchsorted(reference, observed, side="left")
        right = np.searchsorted(reference, observed, side="right")
        # Average the number strictly and weakly above to give equal values an
        # identical mid-rank survival probability.
        exceed = len(reference) - 0.5 * (left + right)
        probability = (exceed + 0.5) / (len(reference) + 1.0)
        result[local] = np.minimum(-np.log(probability), CONFIG["tail_clip"])
    if not np.isfinite(result).all():
        raise ValueError("Invalid Model104 empirical null feature")
    return result


def main():
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, fingerprint = initialize()
        layout = load_development_layout(ROOT)
        _, _, _, _, baseline = probabilities(layout)
        expected = np.asarray(json.loads((PARENT84 / "summary.json").read_text())[
            "candidate_fold_scores"])
        base_scores = fold_scores(layout, baseline)
        np.testing.assert_allclose(base_scores, expected, rtol=0.0, atol=1e-12)
        evidence = np.load(ROOT / "models/model95/real_glr.npy", mmap_mode="r",
                           allow_pickle=False)[:, 0]
        if evidence.shape != (layout.row_count,) or not np.isfinite(evidence).all():
            raise ValueError("Invalid Model104 scale GLR")
        maximum_time = int(layout.time_online.max(initial=0))
        order = np.argsort(layout.time_online, kind="stable")
        sorted_time = layout.time_online[order]
        rows_by_time = []
        for step in range(maximum_time + 1):
            left, right = np.searchsorted(sorted_time, (step, step + 1))
            rows_by_time.append(order[left:right])
        all_folds = np.arange(5)
        candidate = np.empty(layout.row_count, dtype=np.float32)
        records = []
        for outer in range(5):
            development = all_folds[all_folds != outer]
            score_by_weight = np.zeros((len(development), len(CONFIG["weights"])),
                                       dtype=np.float64)
            for row, inner in enumerate(development):
                reference = development[development != inner]
                inner_rows = np.flatnonzero(layout.row_fold == inner)
                transformed = null_tail_feature(
                    layout, evidence, reference, inner_rows, rows_by_time)
                for column, weight in enumerate(CONFIG["weights"]):
                    value = (baseline[inner_rows].astype(np.float64) +
                             weight * transformed).astype(np.float32)
                    score_by_weight[row, column] = tsauc(
                        layout.target[inner_rows], value, layout.time_online[inner_rows])
            means = score_by_weight.mean(axis=0)
            selected = int(np.argmax(means))
            held_rows = np.flatnonzero(layout.row_fold == outer)
            held_feature = null_tail_feature(
                layout, evidence, development, held_rows, rows_by_time)
            candidate[held_rows] = (baseline[held_rows].astype(np.float64) +
                CONFIG["weights"][selected] * held_feature).astype(np.float32)
            records.append({"held_fold": outer, "inner_folds": development.tolist(),
                "inner_scores_by_weight": score_by_weight.tolist(),
                "mean_inner_scores": means.tolist(), "selected_index": selected,
                "selected_weight": CONFIG["weights"][selected],
                "inner_gain": float(means[selected] - base_scores[development].mean()),
                "held_feature_quantiles": {str(q): float(np.quantile(held_feature, q))
                    for q in (0.0, 0.5, 0.9, 0.99, 1.0)}})
            print(f"Model104 held {outer}: selected weight {CONFIG['weights'][selected]:.3f}, "
                  f"nested inner gain={records[-1]['inner_gain']:+.9f}", flush=True)
        if not np.isfinite(candidate).all():
            raise ValueError("Invalid Model104 cross-fitted prediction")
        scores = fold_scores(layout, candidate)
        gains = scores - base_scores
        scorers = []
        for fold in range(5):
            rows = layout.row_fold == fold
            series = np.flatnonzero(layout.inner_series_fold == fold)
            scorers.append((series, [WeightedTSAUC(layout.target[rows], value[rows],
                layout.time_online[rows], layout.series_position[rows])
                for value in (baseline, candidate)]))
        rng, boot = np.random.default_rng(CONFIG["bootstrap_seed"]), []
        for repeat in range(CONFIG["bootstrap_replicates"]):
            local = []
            for series, pair in scorers:
                multiplicity = rng.multinomial(len(series), np.full(len(series), 1.0 / len(series)))
                weights = np.zeros(len(layout.inner_series_fold))
                weights[series] = multiplicity
                local.append([scorer.score(weights) for scorer in pair])
            mean = np.asarray(local).mean(axis=0)
            boot.append(float(mean[1] - mean[0]))
            if (repeat + 1) % 25 == 0:
                commit_checkpoint(directory / "bootstrap.npz", {"gains": np.asarray(boot)},
                    {"protocol_sha256": fingerprint})
        lower = float(np.quantile(boot, (1.0 - CONFIG["confidence"]) / 2.0))
        halves = (layout.series_ids * 0x9E3779B1) & 1
        split = []
        for half in (0, 1):
            rows = halves[layout.series_position] == half
            split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows])
                               - tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
        full_score, gain = float(scores.mean()), float(gains.mean())
        wins = int(np.sum(gains > 0.0))
        passed = bool(full_score >= CONFIG["target_tsauc"] and
            gain >= CONFIG["minimum_full_gain"] and wins >= CONFIG["minimum_fold_wins"] and
            lower > 0.0 and min(split) > 0.0)
        summary = {"model": 104, "selection": records,
            "baseline_fold_scores": base_scores.tolist(), "candidate_fold_scores": scores.tolist(),
            "fold_gains": gains.tolist(), "full_oof_score": full_score,
            "full_oof_gain": gain, "fold_wins": wins, "bootstrap_lower_95": lower,
            "split_half_gains": split, "passes": passed,
            "leakage_audit": "Outer held fold excluded from null tables and nested weight selection.",
            "deployment": "No packaging, upload, or cloud run performed."}
        write_json(directory / "summary.json", summary)
        result = ("# Model104: nested cross-fitted empirical-null scale GLR\n\n"
            "| Arm | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
            "|---|---:|---:|---:|---:|---|\n"
            f"| Model84 | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
            f"| Empirical-null GLR | {full_score:.6f} | {gain:+.6f} | {wins}/5 | "
            f"{lower:+.6f} | {passed} |\n\n"
            f"Selected weights: {[record['selected_weight'] for record in records]}.\n"
            f"Fold gains: {gains.tolist()}. Split-half gains: {split}.\n\n"
            f"{summary['leakage_audit']}\n\n{summary['deployment']}\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
