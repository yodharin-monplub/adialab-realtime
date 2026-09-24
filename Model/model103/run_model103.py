"""Cross-fit two-regime scale-GLR weights over complete held-series folds."""
from __future__ import annotations

import fcntl
import itertools
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
            raise ValueError("Changed Model103 protocol")
        for relative, expected in protocol["contract"]["hashes"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Changed frozen Model103 input: {relative}")
        return directory, fingerprint
    if (sha256(PARENT84 / "protocol.json") != CONFIG["model84_protocol_sha256"] or
            sha256(PARENT95 / "protocol.json") != CONFIG["model95_protocol_sha256"]):
        raise ValueError("Changed Model103 parent")
    paths = [HERE / "run_model103.py", HERE / "prespec.json", HERE / "readme.txt",
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


def candidate_value(baseline, evidence, online, pair):
    weight = np.where(online <= CONFIG["time_cutoff"], pair[0], pair[1])
    return (baseline.astype(np.float64) + weight * evidence).astype(np.float32)


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
            raise ValueError("Invalid Model103 scale GLR")
        pairs = list(itertools.product(CONFIG["weights"], repeat=2))
        score_table = np.empty((5, len(pairs)), dtype=np.float64)
        values = []
        for column, pair in enumerate(pairs):
            value = candidate_value(baseline, evidence, layout.time_online, pair)
            values.append(value)
            score_table[:, column] = fold_scores(layout, value)
            print(f"Model103 pair early={pair[0]:.3f} late={pair[1]:.3f}: "
                  f"fold mean={score_table[:, column].mean():.9f}", flush=True)
        candidate = np.empty(layout.row_count, dtype=np.float32)
        records = []
        for held in range(5):
            training_folds = np.asarray([fold for fold in range(5) if fold != held])
            training_means = score_table[training_folds].mean(axis=0)
            selected = int(np.argmax(training_means))
            rows = layout.row_fold == held
            candidate[rows] = values[selected][rows]
            records.append({"held_fold": held, "selected_index": selected,
                "early_weight": pairs[selected][0], "late_weight": pairs[selected][1],
                "training_fold_mean": float(training_means[selected]),
                "training_gain": float(training_means[selected] - base_scores[training_folds].mean())})
            print(f"Model103 held {held}: selected {pairs[selected]}", flush=True)
        if not np.isfinite(candidate).all():
            raise ValueError("Invalid Model103 cross-fitted prediction")
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
            for series, pair_scorers in scorers:
                multiplicity = rng.multinomial(len(series), np.full(len(series), 1.0 / len(series)))
                weights = np.zeros(len(layout.inner_series_fold))
                weights[series] = multiplicity
                local.append([scorer.score(weights) for scorer in pair_scorers])
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
        summary = {"model": 103, "selection": records,
            "baseline_fold_scores": base_scores.tolist(), "candidate_fold_scores": scores.tolist(),
            "fold_gains": gains.tolist(), "full_oof_score": full_score,
            "full_oof_gain": gain, "fold_wins": wins, "bootstrap_lower_95": lower,
            "split_half_gains": split, "passes": passed,
            "deployment": "No packaging, upload, or cloud run performed."}
        write_json(directory / "summary.json", summary)
        result = ("# Model103: cross-fitted early/late scale-GLR calibration\n\n"
            "| Arm | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
            "|---|---:|---:|---:|---:|---|\n"
            f"| Model84 | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
            f"| Cross-fitted early/late GLR | {full_score:.6f} | {gain:+.6f} | "
            f"{wins}/5 | {lower:+.6f} | {passed} |\n\n"
            f"Selected weights by held fold: {records}.\n\n"
            f"Fold gains: {gains.tolist()}. Split-half gains: {split}.\n\n"
            f"{summary['deployment']}\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
