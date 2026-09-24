"""Confirm Model93's locked direct scale-evidence correction."""
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
PARENT90 = ROOT / CONFIG["model90_run"]
PARENT93 = ROOT / CONFIG["model93_run"]


def initialize():
    if (HERE / "runs/latest.json").exists():
        latest = json.loads((HERE / "runs/latest.json").read_text())
        directory = ROOT / latest["directory"]
        protocol = json.loads((directory / "protocol.json").read_text())
        fingerprint = sha256(directory / "protocol.json")
        if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
            raise ValueError("Changed Model94 protocol")
        for relative, expected in protocol["contract"]["hashes"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Changed frozen Model94 input: {relative}")
        return directory, fingerprint
    if (sha256(PARENT84 / "protocol.json") != CONFIG["model84_protocol_sha256"] or
            sha256(PARENT90 / "protocol.json") != CONFIG["model90_protocol_sha256"] or
            sha256(PARENT93 / "protocol.json") != CONFIG["model93_protocol_sha256"]):
        raise ValueError("Changed Model94 parent")
    pilot = json.loads((PARENT93 / "pilot_decision.json").read_text())
    if (not pilot["continue_confirmation"] or
            pilot["selected_weight"] != CONFIG["selected_weight"]):
        raise ValueError("Model94 does not match the frozen Model93 selection")
    paths = [HERE / "run_model94.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT84 / "protocol.json", PARENT84 / "summary.json",
        PARENT90 / "protocol.json", PARENT90 / "real_evidence.npy",
        PARENT90 / "real_evidence.json", PARENT93 / "protocol.json",
        PARENT93 / "pilot_decision.json", ROOT / "models/model91/run_model91.py",
        ROOT / "models/model89/run_model89.py", ROOT / "models/report_next_models.py",
        ROOT / "models/model10/sources.py", ROOT / "models/model5/artifacts/split_assignments.npz",
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
        evidence = np.load(PARENT90 / "real_evidence.npy", mmap_mode="r", allow_pickle=False)
        candidate = (baseline.astype(np.float64)
                     + CONFIG["selected_weight"] * np.asarray(evidence[:, 0])).astype(np.float32)
        scores = fold_scores(layout, candidate)
        gains = scores - base_scores
        confirmation = np.asarray(CONFIG["confirmation_folds"])
        scorers = []
        for fold in confirmation:
            rows = layout.row_fold == fold
            series = np.flatnonzero(layout.inner_series_fold == fold)
            pair = [WeightedTSAUC(layout.target[rows], value[rows],
                layout.time_online[rows], layout.series_position[rows])
                for value in (baseline, candidate)]
            scorers.append((series, pair))
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
                    {"protocol_sha256": fingerprint,
                     "selected_weight": CONFIG["selected_weight"]})
        lower = float(np.quantile(boot, (1.0 - CONFIG["confidence"]) / 2.0))
        halves = (layout.series_ids * 0x9E3779B1) & 1
        split = []
        for half in (0, 1):
            rows = np.isin(layout.row_fold, confirmation) & (halves[layout.series_position] == half)
            split.append(float(tsauc(layout.target[rows], candidate[rows], layout.time_online[rows])
                               - tsauc(layout.target[rows], baseline[rows], layout.time_online[rows])))
        confirmation_score = float(scores[confirmation].mean())
        full_score = float(scores.mean())
        confirmation_gain = float(gains[confirmation].mean())
        wins = int(np.sum(gains[confirmation] > 0.0))
        passed = bool(confirmation_score >= CONFIG["target_tsauc"] and
            full_score >= CONFIG["target_tsauc"] and
            confirmation_gain >= CONFIG["minimum_confirmation_gain"] and
            wins >= CONFIG["minimum_confirmation_wins"] and
            lower > 0.0 and min(split) > 0.0)
        summary = {"model": 94, "selected_weight": CONFIG["selected_weight"],
            "baseline_fold_scores": base_scores.tolist(),
            "candidate_fold_scores": scores.tolist(), "fold_gains": gains.tolist(),
            "confirmation_score": confirmation_score, "full_oof_score": full_score,
            "confirmation_gain": confirmation_gain, "confirmation_wins": wins,
            "bootstrap_lower_95": lower, "split_half_gains": split, "passes": passed,
            "deployment": "No packaging, upload, or cloud run performed."}
        write_json(directory / "summary.json", summary)
        result = ("# Model94: locked direct scale correction\n\n"
            "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
            "|---|---:|---:|---:|---:|---:|---|\n"
            f"| Model84 | {base_scores[confirmation].mean():.6f} | {base_scores.mean():.6f} | baseline | baseline | baseline | baseline |\n"
            f"| scale weight {CONFIG['selected_weight']:.2f} | {confirmation_score:.6f} | "
            f"{full_score:.6f} | {confirmation_gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
            f"Fold gains: {gains.tolist()}. Split-half gains: {split}.\n\n"
            f"{summary['deployment']}\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
