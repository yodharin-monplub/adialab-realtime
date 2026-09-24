"""Run/resume the frozen five-fold, five-arm Model 18 study (CPU only)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.config import CONFIGS, SEED
from models.model11.features import FEATURE_NAMES, compute_features
from models.model11.pipeline import CACHE, cache_contract, fit_model, sha256, write_json
from models.model18.augmentation import assemble, augmented_weights, real_fit_rows, select_pairs
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
ARMS = ("control", "raw_10", "aligned_10", "raw_25", "aligned_25")
CONFIG = dict(version=1, arms=list(ARMS), folds=5, fit=CONFIGS[11], model_seed=SEED,
              augmentation_seed=20260918, pairs_per_fold=1000, max_b_reuse=5,
              minimum_a_pre_post=32, alignment_pre_window=128, synthetic_rows=500_000,
              synthetic_dtype="float32", bootstrap_replicates=1000, bootstrap_seed=180906,
              false_alarm_thresholds=[.5, .8], minimum_gain=.001, minimum_fold_wins=4,
              simultaneous_confidence=.95, control_tolerance=1e-6,
              weighting="preserve real-only (step,target) weight totals",
              bootstrap="paired complete series within folds; centered maximum across four contrasts")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_arrays(path, **arrays):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def commit_checkpoint(path, arrays, metadata, artifacts=()):
    """Manifest is committed last; incomplete writes are safely recomputed."""
    save_arrays(path, **arrays)
    write_json(path.with_suffix(".json"), dict(metadata=metadata, sha256=sha256(path),
        artifacts={str(p.name): sha256(p) for p in artifacts}))


def read_checkpoint(path, expected):
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        return None
    record = json.loads(manifest.read_text())
    if record["metadata"] != expected or not path.exists() or sha256(path) != record["sha256"]:
        raise ValueError(f"Corrupt or incompatible checkpoint: {path}")
    for name, checksum in record["artifacts"].items():
        item = path.parent / name
        if not item.is_file() or sha256(item) != checksum:
            raise ValueError(f"Corrupt checkpoint artifact: {item}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def load_pool(layout, fold):
    # Restrict parquet reads to fitting-fold break series; never fetch held labels.
    positions = np.flatnonzero((layout.inner_series_fold >= 0)
        & (layout.inner_series_fold != fold) & (layout.tau_index >= 0))
    ids = layout.series_ids[positions]
    frame = pd.read_parquet(ROOT / "train/X_train.parquet", columns=["value", "period"],
                            filters=[("id", "in", ids.tolist())])
    if frame.index.names != ["id", "time"] or not frame.index.is_monotonic_increasing:
        raise ValueError("Donor input must be sorted by (id,time)")
    mapping = dict(zip(ids.tolist(), positions.tolist()))
    pool = {}
    for sid, group in frame.groupby(level="id", sort=False):
        period = group.period.to_numpy()
        value = group.value.to_numpy(dtype=np.float64)
        history, online = value[period == 1], value[period == 2]
        if not np.array_equal(period, np.r_[np.ones(len(history)), np.full(len(online), 2)]):
            raise ValueError("Donor periods are not historical-then-online")
        p = mapping[sid]
        if len(online) != layout.online_lengths[p]:
            raise ValueError("Donor horizon differs from frozen metadata")
        pool[p] = (history, online)
    if len(pool) != len(positions):
        raise ValueError("Missing fitting donor series")
    return pool


def initialize(layout):
    manifest = json.loads((CACHE / "manifest.json").read_text())
    if manifest["contract"] != cache_contract():
        raise ValueError("Existing real feature cache contract mismatch")
    baseline = ROOT / "models/model11/runs/a982309521d5/summary.json"
    files = [*sorted(HERE.glob("*.py")), ROOT / "models/model11/config.py",
             ROOT / "models/model11/pipeline.py", ROOT / "models/model10/sources.py",
             ROOT / "models/report_next_models.py", ROOT / "tsauc.py",
             ROOT / "models/artifacts/core_features.metadata.npz", CACHE / "features.npy", baseline]
    contract = dict(config=CONFIG, features=manifest["contract"],
        file_hashes={str(p.relative_to(ROOT)): sha256(p) for p in files},
        dependencies={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "lightgbm", "scipy")},
        python=sys.version, expected_control_scores=json.loads(baseline.read_text())["raw"]["fold_scores"],
        reserved_holdout="Historically revealed; kept closed in this study", competition_labels_loaded=False)
    identity = digest(contract)[:16]
    directory = HERE / "runs" / identity
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["contract"] != contract or digest(protocol["plans"]) != protocol["plans_sha256"]:
            raise ValueError("Frozen protocol is incompatible or corrupted")
    else:
        plans = []
        for fold in range(5):
            pool = load_pool(layout, fold)
            pairs, stats = select_pairs(pool, layout.tau_index, layout.inner_series_fold,
                fold, CONFIG["augmentation_seed"] + fold)
            if not pairs:
                raise ValueError(f"No eligible composites in fold {fold}")
            for pair in pairs:
                pair["a_id"] = str(layout.series_ids[pair["a"]])
                pair["b_id"] = str(layout.series_ids[pair["b"]])
            total = sum(p["length"] for p in pairs)
            # Independent substream so donor sampling cannot perturb row choices.
            rng = np.random.default_rng(np.random.SeedSequence([20260918 + fold, 1]))
            selected = np.sort(rng.choice(total, min(total, CONFIG["synthetic_rows"]), replace=False))
            fit = real_fit_rows(layout.row_fold, fold)
            plan = dict(fold=fold, pairs=pairs, stats=stats, synthetic_total_rows=total,
                        synthetic_selected_rows=selected.tolist(), real_fit_rows=fit.tolist())
            plans.append(plan)
            print(f"Frozen fold {fold}: {len(pairs)} pairs, {total:,} synthetic rows; {stats}", flush=True)
            del pool
        protocol = dict(contract=contract, plans=plans, plans_sha256=digest(plans))
        write_json(protocol_path, protocol)
    protocol_sha = sha256(protocol_path)
    latest = HERE / "runs/latest.json"
    if latest.exists():
        prior = json.loads(latest.read_text())
        if prior.get("directory") == str(directory.relative_to(ROOT)) and prior["protocol_sha256"] != protocol_sha:
            raise ValueError("Frozen protocol hash differs from latest checkpoint")
    write_json(latest, dict(directory=str(directory.relative_to(ROOT)), protocol_sha256=protocol_sha))
    return directory, protocol, protocol_sha


def synthetic_features(layout, plan, aligned):
    pool = load_pool(layout, plan["fold"])
    selected = np.asarray(plan["synthetic_selected_rows"], dtype=np.int64)
    x = np.empty((len(selected), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(len(selected), dtype=np.uint8)
    steps = np.empty(len(selected), dtype=np.int32)
    offset = 0
    for pair in plan["pairs"]:
        for key in ("a", "b"):
            p = pair[key]
            if layout.inner_series_fold[p] < 0 or layout.inner_series_fold[p] == plan["fold"]:
                raise AssertionError("Held or reserved donor in frozen plan")
        h, o, target, time_online = assemble(pool, pair, aligned)
        # Full causal calculation precedes row sampling. No provenance is passed.
        block = compute_features(h, o)
        left, right = np.searchsorted(selected, [offset, offset + len(o)])
        local = selected[left:right] - offset
        x[left:right], y[left:right], steps[left:right] = block[local], target[local], time_online[local]
        offset += len(o)
    if offset != plan["synthetic_total_rows"] or not np.isfinite(x).all():
        raise AssertionError("Synthetic feature assembly failed")
    return x, y, steps


def checkpoint_meta(protocol_sha, fold, arm):
    return dict(protocol_sha256=protocol_sha, fold=fold, arm=arm)


def validate_predictions(saved, held):
    if (not np.array_equal(saved["held_rows"], held)
            or saved["prediction"].shape != held.shape
            or not np.isfinite(saved["prediction"]).all()
            or np.any((saved["prediction"] < 0) | (saved["prediction"] > 1))):
        raise ValueError("Invalid prediction values or held-row alignment")


def train_one(directory, protocol_sha, layout, x, plan, arm, synthetic=None):
    fold = plan["fold"]
    path = directory / f"{arm}_fold{fold}.npz"
    metadata = checkpoint_meta(protocol_sha, fold, arm)
    held = np.flatnonzero(layout.row_fold == fold)
    saved = read_checkpoint(path, metadata)
    if saved is not None:
        validate_predictions(saved, held)
        return float(tsauc(layout.target[held], saved["prediction"], layout.time_online[held]))
    started = time.monotonic()
    fit = np.asarray(plan["real_fit_rows"], dtype=np.int64)
    np.testing.assert_array_equal(fit, real_fit_rows(layout.row_fold, fold))
    if np.intersect1d(layout.series_position[fit], layout.series_position[held]).size:
        raise AssertionError("Real fit rows leak held series")
    real_x, real_y, real_t = np.asarray(x[fit]), layout.target[fit], layout.time_online[fit]
    print(f"Fitting {arm}, fold {fold}: {len(fit):,} real rows", flush=True)
    if arm == "control":
        model = fit_model(CONFIGS[11], real_x, real_y, real_t, SEED + fold)
        accounting = dict(actual_share=0.)
    else:
        sx, sy, st = synthetic
        rw, sw, accounting = augmented_weights(real_y, real_t, sy, st, int(arm.split("_")[1])/100)
        params = dict(objective="binary", metric="None", verbosity=-1, num_threads=4,
            deterministic=True, force_col_wise=True, seed=SEED + fold,
            feature_fraction=.9, max_bin=127, **CONFIGS[11]["parameters"])
        data = lgb.Dataset(np.concatenate((real_x, sx)), label=np.r_[real_y, sy],
                           weight=np.r_[rw, sw], feature_name=FEATURE_NAMES)
        model = lgb.train(params, data, num_boost_round=CONFIGS[11]["rounds"])
        del data
    prediction = np.empty(len(held), dtype=np.float32)
    for start in range(0, len(held), 100_000):
        prediction[start:start+100_000] = model.predict(x[held[start:start+100_000]], num_threads=4)
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    model_path = directory / f"{arm}_fold{fold}.txt"
    temporary = model_path.with_suffix(".partial.txt")
    model.save_model(str(temporary))
    temporary.replace(model_path)
    details = directory / f"{arm}_fold{fold}.fit.json"
    write_json(details, dict(score=score, elapsed_seconds=time.monotonic()-started,
        real_fit_rows=len(fit), synthetic_fit_rows=0 if synthetic is None else len(synthetic[1]), weights=accounting))
    commit_checkpoint(path, dict(prediction=prediction, held_rows=held), metadata, (model_path, details))
    print(f"Finished {arm}, fold {fold}: TSAUC={score:.9f}, {time.monotonic()-started:.1f}s", flush=True)
    del model, real_x
    gc.collect()
    return float(score)


def run():
    layout = load_development_layout(ROOT)
    directory, protocol, protocol_sha = initialize(layout)
    x = np.load(CACHE / "features.npy", mmap_mode="r")
    if x.shape != (layout.row_count, len(FEATURE_NAMES)) or x.dtype != np.float32:
        raise ValueError("Real feature cache dimensions or dtype mismatch")
    scores = [train_one(directory, protocol_sha, layout, x, plan, "control") for plan in protocol["plans"]]
    expected = protocol["contract"]["expected_control_scores"]
    np.testing.assert_allclose(scores, expected, atol=CONFIG["control_tolerance"], rtol=0,
                               err_msg="Fresh control failed Model 11 reproduction gate")
    print("Fresh control reproduces Model 11; starting four fixed augmentation arms.", flush=True)
    for plan in protocol["plans"]:
        for aligned in (False, True):
            names = [f'{"aligned" if aligned else "raw"}_{share}' for share in (10, 25)]
            pending = [name for name in names if read_checkpoint(directory / f'{name}_fold{plan["fold"]}.npz',
                checkpoint_meta(protocol_sha, plan["fold"], name)) is None]
            if not pending:
                continue
            print(f"Computing {'aligned' if aligned else 'raw'} features, fold {plan['fold']}", flush=True)
            synthetic = synthetic_features(layout, plan, aligned)
            for name in pending:
                train_one(directory, protocol_sha, layout, x, plan, name, synthetic)
            del synthetic
            gc.collect()
    from models.model18.report import report
    report(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    args = parser.parse_args()
    if args.command == "run":
        run()
    else:
        from models.model18.report import report
        report()


if __name__ == "__main__":
    main()
