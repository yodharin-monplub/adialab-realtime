"""Train Model62's learner on a broader frozen aligned-splice plan."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.augmentation import select_pairs
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import build_cache, load_real_pool
from models.model20.run_model20 import synthetic_tasks
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
    for key in ("model20_run", "model20_protocol_sha256", "model57_run",
                "model57_protocol_sha256", "lightgbm", "synthetic_mass"):
        if CONFIG[key] != run_model62.CONFIG[key]:
            raise ValueError(f"Model78 changed fixed Model62 field: {key}")
    parent20 = json.loads((PARENT20 / "protocol.json").read_text())
    return p62, parent20


def make_plans(layout, pool, parent62):
    plans = []
    for fold in range(5):
        donors = {position: values for position, values in pool.items()
                  if layout.inner_series_fold[position] not in (fold, (fold + 1) % 5)}
        pairs, stats = select_pairs(donors, layout.tau_index, layout.inner_series_fold,
            fold, CONFIG["plan_seed"] + fold, count=CONFIG["pair_count"],
            reuse=CONFIG["max_b_reuse"])
        if len(pairs) != CONFIG["pair_count"] or stats["shortfall"]:
            raise ValueError(f"Model78 fold {fold} donor-plan shortfall")
        total = sum(pair["length"] for pair in pairs)
        if total <= CONFIG["synthetic_rows"]:
            raise ValueError("Model78 synthetic pool is not larger than its row sample")
        selected = np.sort(np.random.default_rng(
            np.random.SeedSequence([CONFIG["plan_seed"] + fold, 1])).choice(
                total, CONFIG["synthetic_rows"], replace=False))
        plans.append(dict(fold=fold,
            fit_rows=parent62["plans"][fold]["fit_rows"], pairs=pairs, stats=stats,
            selected=selected.tolist(), synthetic_total=total))
        print(f"Frozen Model78 fold {fold}: {len(pairs)} pairs, {total:,} possible rows", flush=True)
    return plans


def initialize(parent62, parent20, layout, pool):
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    paths = [HERE / "run_model78.py", HERE / "prespec.json", HERE / "readme.txt",
        PARENT20 / "protocol.json", PARENT20 / "real_features.cache.json",
        PARENT57 / "protocol.json", PARENT57 / "real_white.json",
        PARENT62 / "protocol.json", PARENT62 / "summary.json",
        ROOT / CONFIG["model67_run"] / "protocol.json",
        ROOT / CONFIG["model67_run"] / "summary.json",
        ROOT / "models/model18/augmentation.py", ROOT / "models/model20/features.py",
        ROOT / "models/model20/run_model20.py", ROOT / "models/model44/run_model44.py",
        ROOT / "models/model57/run_model57.py", ROOT / "models/model57/whitening.py",
        ROOT / "models/model62/report62.py", ROOT / "models/model67/run_model67.py",
        ROOT / "models/model10/sources.py", ROOT / "models/report_next_models.py",
        ROOT / "tsauc.py", ROOT / "train/X_train.parquet",
        ROOT / "train/y_train_index.parquet",
        ROOT / "models/model5/artifacts/split_assignments.npz"]
    contract = dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in paths},
        names=parent62["contract"]["names"],
        union_names=parent62["contract"]["union_names"],
        base_names=parent20["contract"]["names"],
        dependencies=parent62["contract"]["dependencies"])
    plans = make_plans(layout, pool, parent62)
    protocol = dict(contract=contract, plans=plans, plans_sha256=digest(plans))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Model78 protocol mismatch")
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
        raise ValueError("Changed Model78 protocol")
    if digest(protocol["plans"]) != protocol["plans_sha256"]:
        raise ValueError("Changed Model78 donor plan")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model78 input: {relative}")
    return directory, protocol, fingerprint


def run_fold(directory, protocol, fingerprint, layout, pool, real, fold):
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(directory / f"{ARM}_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm=ARM))
    if saved is not None:
        validate_predictions(saved, held)
        print(f"Model78 fold {fold} already complete", flush=True)
        return
    plan = protocol["plans"][fold]
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    if len(tasks) != CONFIG["pair_count"] or len(sy) != CONFIG["synthetic_rows"]:
        raise ValueError("Changed Model78 synthetic task dimensions")
    base = build_cache(directory, f"broad_aligned_fold{fold}_features", tasks, len(sy),
        protocol["contract"]["base_names"][fold], fingerprint, workers=CONFIG["threads"])
    white = white_source.synthetic_white(directory, fingerprint, fold, tasks, len(sy))
    sx = np.concatenate((np.asarray(base), np.asarray(white)), axis=1)
    del tasks, base, white
    fit_one(directory, protocol, fingerprint, layout, real, plan, "aligned_40", (sx, sy, st))
    del sx, sy, st
    gc.collect()


def new_prediction(layout, directory, fingerprint, folds):
    values = report62.candidate_margins(layout, directory, fingerprint, folds)
    if report62.CHOICES[3] != (0.0, 1.0):
        raise ValueError("Changed direct-voter report choice")
    return np.asarray(values[4], dtype=np.float32)


def candidate_values(layout, directory, fingerprint, folds):
    parents = run_model67.parent_state()
    selected, baseline, _ = run_model67.predictions(layout, parents)
    new = new_prediction(layout, directory, fingerprint, folds)
    values = [baseline]
    for fraction in CONFIG["replacement_fractions"]:
        values.append((baseline + 0.5 * fraction * (new - selected["62"])).astype(np.float32))
    return values


def score_folds(layout, values, folds):
    return np.asarray([[tsauc(layout.target[layout.row_fold == fold],
        value[layout.row_fold == fold], layout.time_online[layout.row_fold == fold])
        for fold in folds] for value in values])


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
        raise ValueError("Changed Model78 pilot decision")
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
    summary = dict(model=78, pilot=decision, baseline_fold_scores=base.tolist(),
        candidate_fold_scores=candidate.tolist(), confirmation_score=confirmation_score,
        full_oof_score=full_score, confirmation_gain=gain, confirmation_wins=wins,
        bootstrap_lower_95=lower, split_half_gains=split, passes=passed,
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory / "summary.json", summary)
    result = ("# Model78: broader aligned-splice donor coverage\n\n"
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
        summary = dict(model=78, pilot=decision, passes=False,
            deployment="No packaging, upload, or cloud run performed.")
        write_json(directory / "summary.json", summary)
        result = ("# Model78: broader aligned-splice donor coverage\n\n"
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
        print(f"Model78 pilot selected replacement fraction {decision['selected_fraction']:.2f}; "
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
        layout = load_development_layout(ROOT)
        pool = load_real_pool(layout)
        directory, protocol, fingerprint = initialize(parent62, parent20, layout, pool)
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
        for fold in CONFIG["pilot_folds"]:
            run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
        if report(directory, fingerprint, layout):
            for fold in CONFIG["confirmation_folds"]:
                run_fold(directory, protocol, fingerprint, layout, pool, real, fold)
            report(directory, fingerprint, layout)


if __name__ == "__main__":
    main()
