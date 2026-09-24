"""Causal reference-versus-online kernel signal and frozen blend evaluation."""
from __future__ import annotations

import argparse
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
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint
from models.model20.features import load_real_pool
from models.model39 import run_model39 as incumbent
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    d39, _, s39 = incumbent.frozen_study()
    if s39 != CONFIG["model39_protocol_sha256"]:
        raise ValueError("Changed Model39 protocol")
    paths = [HERE / "run_model45.py", HERE / "prespec.json", HERE / "readme.txt",
             ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
             d39 / "protocol.json"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        model39_run=str(d39.relative_to(ROOT))))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model45 protocol mismatch")
    if not path.exists():
        write_json(path, protocol)
    fingerprint = sha256(path)
    write_json(HERE / "runs/latest.json", dict(directory=str(directory.relative_to(ROOT)),
        protocol_sha256=fingerprint))
    return directory, protocol, fingerprint


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model45 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model45 input: {relative}")
    return directory, protocol, fingerprint


def _standardize(reference: np.ndarray, online: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = float(np.median(reference))
    mad = float(np.median(np.abs(reference - location))) * 1.4826
    scale = max(mad, float(np.std(reference)) * 0.1, 1e-8)
    bound = CONFIG["clip"]
    return (np.clip((reference - location) / scale, -bound, bound),
            np.clip((online - location) / scale, -bound, bound))


def _window_discrepancy(reference: np.ndarray, online: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(CONFIG["frequencies"], dtype=np.float64)[:, None]
    ref_phase = frequencies * reference[None, :]
    reference_cos = np.cos(ref_phase).mean(axis=1)[:, None]
    reference_sin = np.sin(ref_phase).mean(axis=1)[:, None]
    phase = frequencies * online[None, :]
    cosine = np.c_[np.zeros(len(frequencies)), np.cumsum(np.cos(phase), axis=1)]
    sine = np.c_[np.zeros(len(frequencies)), np.cumsum(np.sin(phase), axis=1)]
    ending = np.arange(1, len(online) + 1)
    output = np.zeros(len(online), dtype=np.float64)
    for width in CONFIG["windows"]:
        length = np.minimum(ending, width)
        start = ending - length
        mean_cos = (cosine[:, ending] - cosine[:, start]) / length
        mean_sin = (sine[:, ending] - sine[:, start]) / length
        distance = np.mean((mean_cos - reference_cos) ** 2 +
                           (mean_sin - reference_sin) ** 2, axis=0)
        effective = len(reference) * length / (len(reference) + length)
        np.maximum(output, effective * distance, out=output)
    return output


def kernel_score(history: np.ndarray, online: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float64)
    online = np.asarray(online, dtype=np.float64)
    if len(history) < 2 or not len(online):
        raise ValueError("Need at least two historical and one online observation")
    if not (np.isfinite(history).all() and np.isfinite(online).all()):
        raise ValueError("Nonfinite series")
    historical_delta = np.diff(history)
    online_delta = np.diff(np.r_[history[-1], online])
    level_ref, level_online = _standardize(history, online)
    delta_ref, delta_online = _standardize(historical_delta, online_delta)
    score = 0.5 * (_window_discrepancy(level_ref, level_online) +
                   _window_discrepancy(delta_ref, delta_online))
    return np.log1p(score).astype(np.float32)


def signal_checkpoint(directory, layout):
    path = directory / "kernel_signal.npy"
    if path.exists():
        signal = np.load(path, mmap_mode="r")
        if signal.shape != (layout.row_count,) or signal.dtype != np.float32:
            raise ValueError("Invalid Model45 signal cache")
        return signal
    pool = load_real_pool(layout)
    signal = np.empty(layout.row_count, dtype=np.float32)
    position = layout.series_position
    starts = np.flatnonzero(np.r_[True, position[1:] != position[:-1]])
    ends = np.r_[starts[1:], layout.row_count]
    for number, (start, end) in enumerate(zip(starts, ends, strict=True), 1):
        series = int(position[start])
        history, online = pool[series]
        if end - start != len(online) or not np.array_equal(
                layout.time_online[start:end], np.arange(len(online))):
            raise ValueError("Model45 series layout mismatch")
        signal[start:end] = kernel_score(history, online)
        if number % 1000 == 0:
            print(f"Model45 analytic signal: {number}/{len(starts)} series", flush=True)
    if len(starts) != layout.series_count or not np.isfinite(signal).all():
        raise ValueError("Invalid Model45 signal")
    temporary = path.with_suffix(".partial.npy")
    with temporary.open("wb") as target:
        np.save(target, signal, allow_pickle=False)
    temporary.replace(path)
    return signal


def report(directory, fingerprint, layout, signal):
    base = incumbent.margins(layout)[2].astype(np.float64)
    pilot_rows = np.isin(layout.row_fold, CONFIG["pilot_folds"])
    signal_quantiles = np.quantile(signal[pilot_rows], [0.25, 0.5, 0.75])
    base_quantiles = np.quantile(base[pilot_rows], [0.25, 0.5, 0.75])
    signal_iqr = signal_quantiles[2] - signal_quantiles[0]
    base_iqr = base_quantiles[2] - base_quantiles[0]
    if signal_iqr <= 0 or base_iqr <= 0:
        raise ValueError("Degenerate Model45 pilot scale")
    scaled = (signal.astype(np.float64) - signal_quantiles[1]) * base_iqr / signal_iqr
    margins = [base] + [base + weight * scaled for weight in CONFIG["blend_weights"]]
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        margin[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in range(5)] for margin in margins])
    pilot = np.asarray(CONFIG["pilot_folds"])
    confirmation = np.asarray(CONFIG["confirmation_folds"])
    pilot_gains = scores[1:, pilot].mean(axis=1) - scores[0, pilot].mean()
    selected_index = int(np.argmax(pilot_gains)) + 1
    selected_weight = CONFIG["blend_weights"][selected_index - 1]
    candidate = scores[selected_index]
    baseline = scores[0]
    gain = float(candidate[confirmation].mean() - baseline[confirmation].mean())
    wins = int(np.sum(candidate[confirmation] > baseline[confirmation]))
    meta = dict(protocol_sha256=fingerprint, selected_weight=selected_weight)
    saved = read_checkpoint(directory / "bootstrap.npz", meta)
    boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for fold in confirmation:
        rows = layout.row_fold == fold
        series = np.flatnonzero(layout.inner_series_fold == fold)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], margin[rows],
            layout.time_online[rows], layout.series_position[rows])
            for margin in (base, margins[selected_index])]))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    completed = len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local = []
        for series, pair in scorers:
            multiplicity = rng.multinomial(len(series),
                np.full(len(series), 1 / len(series)))
            if repeat < completed:
                continue
            weights = np.zeros(len(layout.inner_series_fold))
            weights[series] = multiplicity
            local.append([scorer.score(weights) for scorer in pair])
        if repeat < completed:
            continue
        values = np.asarray(local).mean(axis=0)
        boot.append(float(values[1] - values[0]))
        if (repeat + 1) % 25 == 0:
            commit_checkpoint(directory / "bootstrap.npz",
                dict(gains=np.asarray(boot)), meta)
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    half = (layout.series_ids * 0x9E3779B1) & 1
    confirm_rows = np.isin(layout.row_fold, confirmation)
    split = []
    for value in (0, 1):
        rows = confirm_rows & (half[layout.series_position] == value)
        split.append(float(tsauc(layout.target[rows], margins[selected_index][rows],
            layout.time_online[rows]) - tsauc(layout.target[rows], base[rows],
            layout.time_online[rows])))
    confirm_score = float(candidate[confirmation].mean())
    full_score = float(candidate.mean())
    passed = bool(confirm_score >= CONFIG["target_tsauc"] and
        full_score >= CONFIG["target_tsauc"] and
        gain >= CONFIG["minimum_confirmation_gain"] and
        wins >= CONFIG["minimum_confirmation_wins"] and
        lower > 0 and min(split) > 0)
    summary = dict(model=45, baseline_fold_scores=baseline.tolist(),
        candidate_fold_scores=candidate.tolist(), selected_weight=selected_weight,
        pilot_gains=pilot_gains.tolist(), confirmation_score=confirm_score,
        full_oof_score=full_score, confirmation_gain=gain,
        confirmation_wins=wins, bootstrap_lower_95=lower,
        split_half_gains=split, passes=passed,
        signal_pilot_quantiles=signal_quantiles.tolist(),
        base_pilot_quantiles=base_quantiles.tolist(),
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    lines = ["# Model45: multiscale kernel discrepancy", "",
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| model39 | {baseline[confirmation].mean():.6f} | {baseline.mean():.6f} | +0.000000 | 0/3 | baseline | baseline |",
        f"| kernel weight {selected_weight:.2f} | {confirm_score:.6f} | {full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |",
        "", f"Pilot gains by weight: {dict(zip(CONFIG['blend_weights'], pilot_gains.tolist()))}.",
        f"Confirmation split-half gains: {split}.",
        "These are local held-series OOF scores; prior research reused development folds.",
        "", summary["deployment"], ""]
    result = "\n".join(lines)
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, _, fingerprint = initialize()
        layout = load_development_layout(ROOT)
        signal = signal_checkpoint(directory, layout)
        report(directory, fingerprint, layout, signal)


if __name__ == "__main__":
    main()
