"""Train LightGBM with every row from Model20's frozen splice pairs."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.augmentation import assemble
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import build_cache, load_real_pool
from models.model21.run_model21 import read_cache
from models.model44.run_model44 import fit_one
from models.model57 import run_model57 as white_source
from models.model62 import report62, run_model62
from models.model67 import run_model67
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
PARENT20 = ROOT / CONFIG["model20_run"]
PARENT57 = ROOT / CONFIG["model57_run"]
PARENT62 = ROOT / CONFIG["model62_run"]
ARM = "lightgbm_aligned_40"


def parent_state():
    d62, p62, s62 = run_model62.frozen_study()
    d67, _, s67 = run_model67.frozen_study()
    if d62 != PARENT62 or s62 != CONFIG["model62_protocol_sha256"]:
        raise ValueError("Changed Model62 parent")
    if str(d67.relative_to(ROOT)) != CONFIG["model67_run"] or s67 != CONFIG["model67_protocol_sha256"]:
        raise ValueError("Changed Model67 parent")
    if (CONFIG["model20_run"] != run_model62.CONFIG["model20_run"] or
            CONFIG["model20_protocol_sha256"] != run_model62.CONFIG["model20_protocol_sha256"] or
            CONFIG["model57_run"] != run_model62.CONFIG["model57_run"] or
            CONFIG["model57_protocol_sha256"] != run_model62.CONFIG["model57_protocol_sha256"] or
            CONFIG["lightgbm"] != run_model62.CONFIG["lightgbm"] or
            CONFIG["synthetic_mass"] != run_model62.CONFIG["synthetic_mass"]):
        raise ValueError("Model77 must change only synthetic row coverage")
    parent20 = json.loads((PARENT20 / "protocol.json").read_text())
    return p62, parent20


def initialize(parent62, parent20):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model77.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        PARENT62 / "protocol.json", PARENT62 / "summary.json",
        ROOT / CONFIG["model67_run"] / "protocol.json",
        ROOT / CONFIG["model67_run"] / "summary.json",
        ROOT / "models/model20/features.py", ROOT / "models/model20/run_model20.py",
        ROOT / "models/model44/run_model44.py", ROOT / "models/model57/run_model57.py",
        ROOT / "models/model57/whitening.py", ROOT / "models/model62/report62.py",
        ROOT / "models/model67/run_model67.py", ROOT / "models/model10/sources.py",
        ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
        ROOT / "train/X_train.parquet", ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        names=parent62["contract"]["names"],
        union_names=parent62["contract"]["union_names"],
        base_names=parent20["contract"]["names"],
        dependencies=parent62["contract"]["dependencies"])
    protocol = dict(contract=contract, plans=parent62["plans"],
        plans_sha256=parent62["plans_sha256"])
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model77 protocol mismatch")
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
        raise ValueError("Changed Model77 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model77 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model77 input: {relative}")
    return directory, protocol, fingerprint


def full_synthetic_tasks(pool, plan, layout):
    tasks, targets, times = [], [], []
    total = 0
    for pair in plan["pairs"]:
        for key in ("a", "b"):
            if layout.inner_series_fold[pair[key]] in (-1, plan["fold"], (plan["fold"] + 1) % 5):
                raise ValueError("Synthetic donor leaks held or stopping fold")
        history, online, target, time_online = assemble(pool, pair, True)
        keep = np.arange(len(online), dtype=np.int64)
        tasks.append((history, online, keep))
        targets.append(target)
        times.append(time_online)
        total += len(online)
    if total != plan["synthetic_total"]:
        raise ValueError("Changed full synthetic row accounting")
    return tasks, np.concatenate(targets), np.concatenate(times)


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model77 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    tasks, sy, st = full_synthetic_tasks(pool, plan, layout)
    rows = len(sy)
    if len(tasks) != 1000 or rows != plan["synthetic_total"] or rows <= 400_000:
        raise ValueError("Model77 did not expand synthetic coverage")
    base = build_cache(directory, f"full_aligned_fold{fold}_features", tasks, rows,
        protocol["contract"]["base_names"][fold], fingerprint, workers=CONFIG["threads"])
    white = white_source.synthetic_white(directory, fingerprint, fold, tasks, rows)
    sx = np.concatenate((np.asarray(base), np.asarray(white)), axis=1)
    del tasks, base, white
    fit_one(directory, protocol, fingerprint, layout, real, plan, "aligned_40", (sx, sy, st))
    del sx, sy, st
    gc.collect()


def parent_predictions(layout):
    parents = run_model67.parent_state()
    selected, baseline, _ = run_model67.predictions(layout, parents)
    return selected, baseline


def new_prediction(layout, directory, fingerprint, folds):
    values = report62.candidate_margins(layout, directory, fingerprint, folds)
    # Model62 choice index 3 is (old_fraction=0, voter_weight=1): the new voter.
    if report62.CHOICES[3] != (0.0, 1.0):
        raise ValueError("Changed direct-voter report choice")
    return np.asarray(values[4], dtype=np.float32)


def score_folds(layout, values, folds):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in folds] for value in values])


def candidate_values(layout, directory, fingerprint, folds):
    selected, baseline = parent_predictions(layout)
    new = new_prediction(layout, directory, fingerprint, folds)
    values = [baseline]
    for fraction in CONFIG["replacement_fractions"]:
        # Model62 owns 50% of Model67; replace the frozen fraction of that allocation.
        values.append((baseline + 0.5 * fraction * (new - selected["62"])).astype(np.float32))
    return values


def pilot_decision(directory, fingerprint, layout):
    folds = CONFIG["pilot_folds"]
    values = candidate_values(layout, directory, fingerprint, folds)
    scores = score_folds(layout, values, folds)
    gains = scores[1:].mean(axis=1) - scores[0].mean()
    selected = int(np.argmax(gains))
    decision = dict(protocol_sha256=fingerprint, fractions=CONFIG["replacement_fractions"],
        pilot_gains=gains.tolist(), baseline_scores=scores[0].tolist(),
        candidate_scores=scores[selected + 1].tolist(), selected_index=selected,
        selected_fraction=CONFIG["replacement_fractions"][selected],
        continue_training=bool(gains[selected] >= CONFIG["pilot_minimum_gain"]))
    path = directory / "pilot_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise ValueError("Changed Model77 pilot decision")
    if not path.exists():
        write_json(path, decision)
    return decision


def final_report(directory, fingerprint, layout, decision):
    values = candidate_values(layout, directory, fingerprint, range(5))
    scores = score_folds(layout, values, range(5))
    base = scores[0]
    candidate = scores[decision["selected_index"] + 1]
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
                dict(protocol_sha256=fingerprint, selected_index=decision["selected_index"]))
    lower = float(np.quantile(boot, (1 - CONFIG["confidence"]) / 2))
    halves = (layout.series_ids * 0x9E3779B1) & 1
    split = []
    for half in (0, 1):
        rows = np.isin(layout.row_fold, confirm) & (halves[layout.series_position] == half)
        split.append(float(tsauc(layout.target[rows], values[decision["selected_index"] + 1][rows],
            layout.time_online[rows]) - tsauc(layout.target[rows], values[0][rows],
            layout.time_online[rows])))
    confirmation_score = float(candidate[confirm].mean())
    full_score = float(candidate.mean())
    passed = bool(confirmation_score >= CONFIG["target_tsauc"] and full_score >= CONFIG["target_tsauc"]
        and gain >= CONFIG["minimum_confirmation_gain"] and wins >= CONFIG["minimum_confirmation_wins"]
        and lower > 0 and min(split) > 0)
    summary = dict(model=77, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model77: full aligned-splice coverage\n\n"
        "| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        f"| Model67 | {base[confirm].mean():.6f} | {base.mean():.6f} | baseline | baseline | baseline | baseline |\n"
        f"| replace {decision['selected_fraction']:.2f} of Model62 allocation | {confirmation_score:.6f} | "
        f"{full_score:.6f} | {gain:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\n"
        f"Pilot gains: {dict(zip(CONFIG['replacement_fractions'], decision['pilot_gains']))}.\n"
        f"Confirmation split-half gains: {split}.\n\n{summary['deployment']}\n")
    (directory / "RESULTS.md").write_text(result)
    (HERE / "RESULTS.md").write_text(result)
    print(result, flush=True)


def report(directory, fingerprint, layout):
    decision = pilot_decision(directory, fingerprint, layout)
    if not decision["continue_training"]:
        summary = dict(model=77, pilot=decision, passes=False,
            deployment="No packaging, upload, or cloud run performed.")
        write_json(directory / "summary.json", summary)
        result = ("# Model77: full aligned-splice coverage\n\n"
            f"Pilot gains: {dict(zip(CONFIG['replacement_fractions'], decision['pilot_gains']))}.\n"
            "No choice met the prespecified +0.0001 pilot gate; confirmation folds were not trained.\n"
            "Target 0.640 not reached. No packaging, upload, or cloud run performed.\n")
        (directory / "RESULTS.md").write_text(result)
        (HERE / "RESULTS.md").write_text(result)
        print(result, flush=True)
        return False
    if all((directory / f"{ARM}_fold{fold}.npz").exists()
           for fold in CONFIG["confirmation_folds"]):
        final_report(directory, fingerprint, layout, decision)
    else:
        print(f"Model77 pilot selected replacement fraction {decision['selected_fraction']:.2f}; "
              "confirmation fits pending", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent62, parent20 = parent_state()
        directory, protocol, fingerprint = initialize(parent62, parent20)
        layout = load_development_layout(ROOT)
        if args.command == "report":
            report(directory, fingerprint, layout)
            return
        base_real = read_cache(PARENT20, "real_features", dict(
            protocol_sha256=CONFIG["model20_protocol_sha256"], rows=layout.row_count,
            names=parent20["contract"]["union_names"],
            tasks=int(np.sum(layout.inner_series_fold >= 0))))
        white_real = white_source.cached_white(PARENT57 / "real_white.npy",
            CONFIG["model57_protocol_sha256"], layout.row_count)
        if white_real is None:
            raise ValueError("Missing Model57 real whitening cache")
        real = white_source.CombinedFeatures(base_real, white_real)
        pool = load_real_pool(layout)
        for fold in CONFIG["pilot_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
