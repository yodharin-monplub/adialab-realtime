"""Leakage-controlled joint LightGBM and global feature-selection study.

This driver combines the aligned core, candidate, and novel feature caches.
Complete series are assigned to a fresh deterministic 60/20/20 split.  The
test role cannot be scored until a hash-protected development contract exists.

Typical execution::

    python3 joint_lightgbm_experiment.py development
    python3 joint_lightgbm_experiment.py evaluate-test

``development`` is resumable and performs train-only repeated importance,
coarse subset screening, high-fidelity subset comparison, LightGBM tuning,
and a same-split two-model incumbent. ``evaluate-test`` refits the frozen
design on train+validation and opens test labels once.  No protected archive,
competition test data, or pretrained model is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np

from feature_pipeline import feature_names, load_feature_cache
from train_cv import (
    DEFAULT_MODEL_PARAMETERS,
    deterministic_sample_mask,
    make_series_folds,
    make_tsauc_evaluator,
    pair_weights_from_reference,
)
from tsauc import tsauc


PROJECT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT / "models" / "artifacts" / "joint_lightgbm_v1"
EXPERIMENT_VERSION = "joint-lightgbm-v1"
# Revision 1 was frozen accidentally by an earlier tuning process whose model
# seed changed with every hyperparameter candidate.  Its test was never opened.
# Keep those audit files intact and give the corrected, fixed-seed protocol its
# own immutable namespace so stale and valid contracts cannot be confused.
CONTRACT_REVISION = "fixedseed-v2"
FROZEN_CONTRACT_FILE = "frozen_contract_fixedseed_v2.json"
SELECTED_FEATURE_FILE = "selected_joint_features_fixedseed_v2.npz"
SEALED_RESULT_FILE = "sealed_test_results_fixedseed_v2.json"
SEALED_SENTINEL_FILE = "SEALED_TEST_OPENED_FIXEDSEED_V2.json"
SEALED_MODEL_DIRECTORY = "sealed_test_models_fixedseed_v2"
SEALED_PREDICTION_DIRECTORY = "sealed_test_predictions_fixedseed_v2"
CACHE_NAMES = ("core_features", "candidate_features", "novel_features")
FAMILY_NAMES = ("core", "candidate", "novel")
FOLD_SEED = 20_260_821
TRAIN_FOLDS = (2, 3, 4)
VALID_FOLD = 1
TEST_FOLD = 0
SAMPLE_SEED = 99_173
SCREEN_FRACTION = 0.08
COARSE_FRACTION = 0.075
FULL_FRACTION = 0.20
MODEL_SEED = 61_413
SCREEN_SEEDS = (71_011, 71_029, 71_047)
FEATURE_COUNTS = (128, 256, 384, 512, 768, 1024, 1536)
PREDICTION_CHUNK_ROWS = 30_000
MATERIALIZE_CHUNK_ROWS = 10_000

TUNING_OVERRIDES: tuple[tuple[str, dict[str, object]], ...] = (
    ("default", {}),
    ("leaves15_dense", {"num_leaves": 15, "min_data_in_leaf": 150,
                         "feature_fraction": 0.80, "learning_rate": 0.04,
                         "lambda_l2": 2.0}),
    ("leaves31_dense", {"num_leaves": 31, "min_data_in_leaf": 250,
                         "feature_fraction": 0.80, "learning_rate": 0.03}),
    ("leaves47", {"num_leaves": 47, "min_data_in_leaf": 350,
                   "feature_fraction": 0.65, "learning_rate": 0.04,
                   "lambda_l2": 5.0}),
    ("leaves63", {"num_leaves": 63, "min_data_in_leaf": 400,
                   "feature_fraction": 0.55, "learning_rate": 0.03,
                   "lambda_l2": 5.0}),
    ("leaves63_sparse", {"num_leaves": 63, "min_data_in_leaf": 600,
                          "feature_fraction": 0.40, "learning_rate": 0.04,
                          "lambda_l1": 0.5, "lambda_l2": 8.0}),
    ("leaves95", {"num_leaves": 95, "min_data_in_leaf": 600,
                   "feature_fraction": 0.50, "learning_rate": 0.025,
                   "lambda_l2": 8.0}),
    ("leaves127", {"num_leaves": 127, "min_data_in_leaf": 750,
                    "feature_fraction": 0.45, "learning_rate": 0.025,
                    "lambda_l1": 0.5, "lambda_l2": 10.0}),
    ("fast31", {"num_leaves": 31, "min_data_in_leaf": 300,
                 "feature_fraction": 0.55, "learning_rate": 0.05,
                 "bagging_fraction": 0.75}),
    ("regularized31", {"num_leaves": 31, "min_data_in_leaf": 500,
                        "feature_fraction": 0.70, "learning_rate": 0.035,
                        "lambda_l1": 1.0, "lambda_l2": 12.0}),
)


@dataclass
class ExperimentData:
    caches: list[object]
    matrices: list[np.memmap]
    metadata: dict[str, np.ndarray]
    widths: tuple[int, ...]
    offsets: tuple[int, ...]
    names: list[str]
    series_fold: np.ndarray
    row_fold: np.ndarray

    @property
    def columns(self) -> int:
        return len(self.names)

    @property
    def train_full(self) -> np.ndarray:
        return np.isin(self.row_fold, TRAIN_FOLDS)

    @property
    def valid_full(self) -> np.ndarray:
        return self.row_fold == VALID_FOLD

    @property
    def development_full(self) -> np.ndarray:
        return self.row_fold != TEST_FOLD

    @property
    def test_full(self) -> np.ndarray:
        return self.row_fold == TEST_FOLD


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha_json(value: object) -> str:
    payload = json.dumps(_json_ready(value), sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _column_hash(columns: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(columns, dtype=np.int32).tobytes()).hexdigest()


def _safe(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name)


def _load_data() -> ExperimentData:
    caches = [load_feature_cache(PROJECT / "models" / "artifacts", name)
              for name in CACHE_NAMES]
    metadata = caches[0].load_metadata()
    for cache in caches[1:]:
        candidate = cache.load_metadata()
        for key in ("row_id", "series_position", "time_online", "target",
                    "series_ids", "online_lengths", "tau_index"):
            if not np.array_equal(metadata[key], candidate[key]):
                raise ValueError(f"cache {cache.matrix_path} is misaligned at {key}")
    widths = tuple(cache.columns for cache in caches)
    offsets = tuple(np.cumsum((0, *widths[:-1])).astype(int).tolist())
    names = [name for cache in caches for name in feature_names(cache.banks)]
    if len(names) != sum(widths) or len(names) != len(set(names)):
        raise ValueError("joint feature names are missing or duplicated")
    series_fold = make_series_folds(
        metadata["online_lengths"], metadata["tau_index"], 5, FOLD_SEED
    )
    row_fold = series_fold[metadata["series_position"]]
    data = ExperimentData(
        caches=caches,
        matrices=[cache.open("r") for cache in caches],
        metadata=metadata,
        widths=widths,
        offsets=offsets,
        names=names,
        series_fold=series_fold,
        row_fold=row_fold,
    )
    _write_or_validate_split(data)
    return data


def _write_or_validate_split(data: ExperimentData) -> None:
    roles = {
        "train": np.isin(data.series_fold, TRAIN_FOLDS),
        "validation": data.series_fold == VALID_FOLD,
        "test": data.series_fold == TEST_FOLD,
    }
    if any(np.any(roles[a] & roles[b])
           for a, b in (("train", "validation"), ("train", "test"),
                        ("validation", "test"))):
        raise AssertionError("series split roles overlap")
    manifests = [json.loads(cache.manifest_path.read_text()) for cache in data.caches]
    contract = {
        "experiment_version": EXPERIMENT_VERSION,
        "fold_seed": FOLD_SEED,
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "sealed_test_fold": TEST_FOLD,
        "cache_names": list(CACHE_NAMES),
        "cache_feature_signatures": [item["feature_signature"] for item in manifests],
        "feature_counts": dict(zip(FAMILY_NAMES, data.widths)),
        "summary": {
            role: {
                "series": int(mask.sum()),
                "rows": int(data.metadata["online_lengths"][mask].sum()),
                "broken_series": int(np.sum(data.metadata["tau_index"][mask] >= 0)),
            }
            for role, mask in roles.items()
        },
        "leakage_rule": (
            "Feature generation is causal. Rankings use train only; all design "
            "choices use train/validation only; test labels open after freeze."
        ),
    }
    path = ARTIFACT_ROOT / "split_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("saved split contract differs from current code/data")
    if not path.exists():
        _json_dump(path, contract)


def _parameters(seed: int, overrides: dict[str, object] | None = None) -> dict[str, object]:
    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    parameters.update(overrides or {})
    parameters.update({
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": seed,
        "feature_fraction_seed": seed + 100,
        "bagging_seed": seed + 200,
        "data_random_seed": seed + 300,
    })
    return parameters


def _sample_mask(data: ExperimentData, fraction: float) -> np.ndarray:
    return deterministic_sample_mask(
        data.metadata["row_id"], data.metadata["time_online"], fraction, SAMPLE_SEED
    )


def _materialize(data: ExperimentData, rows: np.ndarray,
                 columns: np.ndarray) -> np.ndarray:
    """Read only selected global columns while bounding transient cache copies."""

    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int32)
    output = np.empty((rows.size, columns.size), dtype=np.float32)
    for matrix, offset, width in zip(data.matrices, data.offsets, data.widths):
        positions = np.flatnonzero((columns >= offset) & (columns < offset + width))
        if positions.size == 0:
            continue
        local = columns[positions] - offset
        for left in range(0, rows.size, MATERIALIZE_CHUNK_ROWS):
            right = min(left + MATERIALIZE_CHUNK_ROWS, rows.size)
            block = np.asarray(matrix[rows[left:right]], dtype=np.float32)
            output[left:right, positions] = block[:, local]
    return np.ascontiguousarray(output)


def _predict(booster: lgb.Booster, data: ExperimentData, rows: np.ndarray,
             columns: np.ndarray) -> np.ndarray:
    output = np.empty(rows.size, dtype=np.float32)
    iteration = booster.best_iteration or booster.current_iteration()
    for left in range(0, rows.size, PREDICTION_CHUNK_ROWS):
        right = min(left + PREDICTION_CHUNK_ROWS, rows.size)
        matrix = _materialize(data, rows[left:right], columns)
        output[left:right] = booster.predict(
            matrix, num_iteration=iteration, num_threads=8
        ).astype(np.float32)
    return output


def _save_selection(name: str, columns: np.ndarray, data: ExperimentData,
                    metadata: dict[str, object]) -> Path:
    path = ARTIFACT_ROOT / "selections" / f"{_safe(name)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, columns=np.asarray(columns, dtype=np.int32),
                        names=np.asarray([data.names[int(i)] for i in columns]),
                        metadata=np.asarray(json.dumps(_json_ready(metadata))))
    return path


def _load_selection(name: str) -> tuple[np.ndarray, list[str], dict[str, object]]:
    path = ARTIFACT_ROOT / "selections" / f"{_safe(name)}.npz"
    with np.load(path, allow_pickle=False) as payload:
        return (payload["columns"].astype(np.int32), payload["names"].tolist(),
                json.loads(str(payload["metadata"])))


def _fit_validation_job(data: ExperimentData, tag: str, columns: np.ndarray,
                        parameters: dict[str, object], sample_fraction: float,
                        max_rounds: int, stopping_rounds: int) -> dict[str, object]:
    directory = ARTIFACT_ROOT / "jobs" / _safe(tag)
    model_path = directory / "model.txt"
    prediction_path = directory / "validation_prediction.f32.npy"
    record_path = directory / "record.json"
    contract = {
        "tag": tag, "column_hash": _column_hash(columns),
        "feature_count": int(columns.size), "parameters": parameters,
        "sample_fraction": sample_fraction, "sample_seed": SAMPLE_SEED,
        "max_rounds": max_rounds, "stopping_rounds": stopping_rounds,
        "split_seed": FOLD_SEED,
    }
    contract_hash = _sha_json(contract)
    if record_path.exists() and model_path.exists() and prediction_path.exists():
        record = json.loads(record_path.read_text())
        if record["contract_hash"] != contract_hash:
            raise ValueError(f"job directory collision for {tag}")
        print(f"{tag}: cached validation TSAUC={record['validation_tsauc']:.6f}",
              flush=True)
        return record

    directory.mkdir(parents=True, exist_ok=True)
    sampled = _sample_mask(data, sample_fraction)
    train_rows = np.flatnonzero(data.train_full & sampled)
    stop_rows = np.flatnonzero(data.valid_full & sampled)
    valid_rows = np.flatnonzero(data.valid_full)
    print(f"{tag}: {train_rows.size:,} train + {stop_rows.size:,} stop rows x "
          f"{columns.size:,} features", flush=True)
    started = time.perf_counter()
    x_train = _materialize(data, train_rows, columns)
    x_stop = _materialize(data, stop_rows, columns)
    y_train = data.metadata["target"][train_rows]
    y_stop = data.metadata["target"][stop_rows]
    train_weight = pair_weights_from_reference(
        y_train, data.metadata["time_online"][train_rows],
        data.metadata["target"][data.train_full],
        data.metadata["time_online"][data.train_full],
    )
    stop_weight = pair_weights_from_reference(
        y_stop, data.metadata["time_online"][stop_rows],
        data.metadata["target"][data.valid_full],
        data.metadata["time_online"][data.valid_full],
    )
    names = [data.names[int(index)] for index in columns]
    train_set = lgb.Dataset(x_train, label=y_train, weight=train_weight,
                            feature_name=names, params=parameters,
                            free_raw_data=True)
    stop_set = lgb.Dataset(x_stop, label=y_stop, weight=stop_weight,
                           feature_name=names, reference=train_set,
                           params=parameters, free_raw_data=True)
    train_set.construct(); stop_set.construct()
    del x_train, x_stop, y_train, y_stop, train_weight, stop_weight
    gc.collect()
    booster = lgb.train(
        parameters, train_set, num_boost_round=max_rounds,
        valid_sets=[stop_set], valid_names=["validation_sample"],
        feval=make_tsauc_evaluator(data.metadata["time_online"][stop_rows]),
        callbacks=[lgb.early_stopping(stopping_rounds, first_metric_only=True,
                                      verbose=False),
                   lgb.log_evaluation(period=100)],
    )
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    del train_set, stop_set
    gc.collect()
    prediction = _predict(booster, data, valid_rows, columns)
    score = tsauc(data.metadata["target"][valid_rows], prediction,
                   data.metadata["time_online"][valid_rows])
    np.save(prediction_path, prediction)
    record = {
        **contract, "contract_hash": contract_hash,
        "validation_tsauc": float(score),
        "best_iteration": int(booster.best_iteration),
        "train_sample_rows": int(train_rows.size),
        "validation_rows": int(valid_rows.size),
        "elapsed_seconds": float(time.perf_counter() - started),
        "model_path": str(model_path),
        "validation_prediction_path": str(prediction_path),
    }
    _json_dump(record_path, record)
    print(f"{tag}: TSAUC={score:.6f}, iteration={booster.best_iteration}, "
          f"elapsed={record['elapsed_seconds'] / 60:.1f} min", flush=True)
    del booster, prediction
    gc.collect()
    return record


def _rank_descending(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=-np.inf, neginf=-np.inf,
                           posinf=np.finfo(np.float64).max)
    return np.lexsort((np.arange(values.size), -values)).astype(np.int32)


def _rank_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(order.size)
    positions[order] = np.arange(order.size)
    return positions


def screen_rankings() -> dict[str, np.ndarray]:
    """Fit three train-only screens and derive stable global rankings."""

    data = _load_data()
    ranking_path = ARTIFACT_ROOT / "rankings" / "rankings.npz"
    summary_path = ARTIFACT_ROOT / "rankings" / "summary.json"
    if ranking_path.exists() and summary_path.exists():
        with np.load(ranking_path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    columns = np.arange(data.columns, dtype=np.int32)
    sampled = _sample_mask(data, SCREEN_FRACTION)
    train_rows = np.flatnonzero(data.train_full & sampled)
    print(f"screen: materializing {train_rows.size:,} train-only rows x "
          f"{data.columns:,} features", flush=True)
    matrix = _materialize(data, train_rows, columns)
    target = data.metadata["target"][train_rows]
    weights = pair_weights_from_reference(
        target, data.metadata["time_online"][train_rows],
        data.metadata["target"][data.train_full],
        data.metadata["time_online"][data.train_full],
    )
    base_parameters = _parameters(SCREEN_SEEDS[0], {
        "learning_rate": 0.05, "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
    })
    dataset = lgb.Dataset(matrix, label=target, weight=weights,
                          feature_name=data.names, params=base_parameters,
                          free_raw_data=True)
    dataset.construct()
    del matrix, target, weights
    gc.collect()

    gains, splits = [], []
    model_directory = ARTIFACT_ROOT / "rankings" / "screen_models"
    model_directory.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(SCREEN_SEEDS):
        model_path = model_directory / f"seed_{seed}.txt"
        if model_path.exists():
            booster = lgb.Booster(model_file=str(model_path))
        else:
            parameters = _parameters(seed, {
                "learning_rate": 0.05, "feature_fraction": 0.70,
                "bagging_fraction": 0.80,
            })
            # A constructed LightGBM Dataset has immutable binning. Keep that
            # seed fixed while varying tree, feature, and bagging randomness.
            parameters["data_random_seed"] = SCREEN_SEEDS[0] + 300
            print(f"screen: seed {seed}, 96 fixed rounds", flush=True)
            booster = lgb.train(parameters, dataset, num_boost_round=96)
            booster.save_model(model_path)
        gains.append(booster.feature_importance("gain").astype(np.float64))
        splits.append(booster.feature_importance("split").astype(np.float64))
        del booster
    del dataset
    gc.collect()
    gain = np.vstack(gains)
    split = np.vstack(splits)
    gain_mean = gain.mean(axis=0)
    split_mean = split.mean(axis=0)
    gain_order = _rank_descending(gain_mean)
    rank_sum = np.zeros(data.columns)
    for row in (*gain, *split):
        rank_sum += _rank_positions(_rank_descending(row))
    consensus_order = np.lexsort((np.arange(data.columns), rank_sum)).astype(np.int32)

    pool = gain_order[:min(1536, data.columns)]
    rng = np.random.default_rng(81_281)
    available = np.flatnonzero(data.train_full)
    correlation_rows = np.sort(rng.choice(
        available, size=min(4096, available.size), replace=False
    ))
    correlation_matrix = _materialize(data, correlation_rows, pool).astype(np.float64)
    correlation_matrix -= correlation_matrix.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(correlation_matrix * correlation_matrix, axis=0))
    scale[~np.isfinite(scale) | (scale < 1e-12)] = np.inf
    correlation_matrix /= scale
    correlation = np.abs(correlation_matrix.T @ correlation_matrix)
    np.fill_diagonal(correlation, 0.0)
    kept: list[int] = []
    for candidate in range(pool.size):
        if not kept or float(np.max(correlation[candidate, kept])) < 0.98:
            kept.append(candidate)
    correlation_order = pool[np.asarray(kept, dtype=np.int32)]
    del correlation_matrix, correlation
    gc.collect()

    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ranking_path, gain=gain, split=split, gain_mean=gain_mean,
        split_mean=split_mean, gain_order=gain_order,
        consensus_order=consensus_order, correlation_order=correlation_order,
    )
    family_ranges = {
        name: [int(offset), int(offset + width)]
        for name, offset, width in zip(FAMILY_NAMES, data.offsets, data.widths)
    }
    top_512 = consensus_order[:512]
    _json_dump(summary_path, {
        "all_rankings_use_train_only": True,
        "screen_seeds": SCREEN_SEEDS,
        "screen_fraction": SCREEN_FRACTION,
        "screen_rows": train_rows.size,
        "feature_count": data.columns,
        "family_ranges": family_ranges,
        "correlation_pool": pool.size,
        "correlation_kept_abs_lt_0p98": correlation_order.size,
        "top_512_family_counts": {
            family: int(np.sum((top_512 >= offset) & (top_512 < offset + width)))
            for family, offset, width in zip(FAMILY_NAMES, data.offsets, data.widths)
        },
    })
    return {
        "gain": gain, "split": split, "gain_mean": gain_mean,
        "split_mean": split_mean, "gain_order": gain_order,
        "consensus_order": consensus_order,
        "correlation_order": correlation_order,
    }


def _prior_union(data: ExperimentData) -> np.ndarray:
    core_path = PROJECT / "models/artifacts/feature_selection_v1/core/selections/consensus_top_768.npz"
    candidate_path = PROJECT / "models/artifacts/feature_selection_v1/candidate/selections/gain_corr_0p98.npz"
    with np.load(core_path, allow_pickle=False) as payload:
        core = payload["columns"].astype(np.int32)
    with np.load(candidate_path, allow_pickle=False) as payload:
        candidate = payload["columns"].astype(np.int32) + data.widths[0]
    return np.concatenate((core, candidate)).astype(np.int32)


def _balanced_columns(data: ExperimentData, order: np.ndarray,
                      count: int) -> np.ndarray:
    """Guarantee 10% novel features and split the remainder by family size."""

    novel_count = min(data.widths[2], max(1, int(round(0.10 * count))))
    remaining = count - novel_count
    core_count = int(round(remaining * data.widths[0] /
                           (data.widths[0] + data.widths[1])))
    quotas = (core_count, remaining - core_count, novel_count)
    selected: list[int] = []
    for offset, width, quota in zip(data.offsets, data.widths, quotas):
        family_order = order[(order >= offset) & (order < offset + width)]
        selected.extend(family_order[:quota].tolist())
    return np.asarray(selected, dtype=np.int32)


def _candidate_selections(data: ExperimentData,
                          rankings: dict[str, np.ndarray]) -> list[dict[str, object]]:
    existing_count = data.widths[0] + data.widths[1]
    candidates: list[dict[str, object]] = []

    def add(name: str, columns: np.ndarray, strategy: str,
            eligible: bool, note: str) -> None:
        columns = np.asarray(columns, dtype=np.int32)
        candidates.append({
            "name": name, "columns": columns, "feature_count": int(columns.size),
            "strategy": strategy, "deployment_eligible": eligible, "note": note,
        })

    add("expanded_all", np.arange(data.columns), "all", True,
        "All core+candidate+novel features; no label-based filtering.")
    add("existing_all", np.arange(existing_count), "all", True,
        "Core+candidate joint model without the new d60 bank.")
    add("prior_union_1439", _prior_union(data), "prior_union", False,
        "Historical selection; ablation only because it predates this split.")
    add("novel_only", np.arange(existing_count, data.columns), "diagnostic", False,
        "Standalone incremental-signal diagnostic; not a deployment candidate.")
    for count in FEATURE_COUNTS:
        if count >= data.columns:
            continue
        add(f"gain_top_{count}", rankings["gain_order"][:count], "gain", True,
            "Repeated train-only LightGBM gain ranking.")
        add(f"consensus_top_{count}", rankings["consensus_order"][:count],
            "consensus", True, "Consensus of gain/split ranks across three seeds.")
        add(f"balanced_top_{count}",
            _balanced_columns(data, rankings["consensus_order"], count),
            "balanced", True, "Family quotas with 10% reserved for novel features.")
        if rankings["correlation_order"].size >= count:
            add(f"gain_corr_top_{count}", rankings["correlation_order"][:count],
                "gain_corr", True, "Gain order after train-only abs-correlation pruning.")
    rng = np.random.default_rng(77_391)
    add("random_control_512", np.sort(rng.choice(data.columns, 512, replace=False)),
        "negative_control", False, "Fixed negative control; never deployable.")
    output, seen = [], set()
    for candidate in candidates:
        digest = _column_hash(candidate["columns"])
        if digest in seen:
            continue
        seen.add(digest)
        candidate["column_hash"] = digest
        output.append(candidate)
    return output


def run_selection() -> dict[str, object]:
    """Use successive fidelity to compare global feature-selection contracts."""

    data = _load_data()
    rankings = screen_rankings()
    candidates = _candidate_selections(data, rankings)
    coarse_records = []
    for candidate in candidates:
        columns = candidate["columns"]
        _save_selection(candidate["name"], columns, data, {
            key: value for key, value in candidate.items() if key != "columns"
        })
        record = _fit_validation_job(
            data, f"coarse__{candidate['name']}", columns,
            _parameters(MODEL_SEED), COARSE_FRACTION, 260, 35,
        )
        coarse_records.append({
            **{key: value for key, value in candidate.items() if key != "columns"},
            "validation_tsauc": record["validation_tsauc"],
            "best_iteration": record["best_iteration"],
        })

    # Always retain the two all-feature controls and historical union. Add the
    # best member of each eligible strategy plus the three global leaders.
    by_name = {candidate["name"]: candidate for candidate in candidates}
    shortlist_names = {"expanded_all", "existing_all", "prior_union_1439"}
    eligible = [row for row in coarse_records if row["deployment_eligible"]]
    for strategy in ("gain", "consensus", "balanced", "gain_corr"):
        rows = [row for row in eligible if row["strategy"] == strategy]
        if rows:
            shortlist_names.add(max(rows, key=lambda row: row["validation_tsauc"])["name"])
    for row in sorted(eligible, key=lambda item: -item["validation_tsauc"])[:3]:
        shortlist_names.add(row["name"])

    full_records = []
    for name in sorted(shortlist_names):
        candidate = by_name[name]
        record = _fit_validation_job(
            data, f"full__{name}", candidate["columns"],
            _parameters(MODEL_SEED + 100), FULL_FRACTION, 700, 70,
        )
        full_records.append({
            **{key: value for key, value in candidate.items() if key != "columns"},
            "validation_tsauc": record["validation_tsauc"],
            "best_iteration": record["best_iteration"],
            "job_tag": f"full__{name}",
        })
    primary = max(
        (row for row in full_records if row["deployment_eligible"]),
        key=lambda row: (row["validation_tsauc"], -row["feature_count"]),
    )
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "coarse_fraction": COARSE_FRACTION,
        "full_fraction": FULL_FRACTION,
        "coarse_results": coarse_records,
        "shortlist": sorted(shortlist_names),
        "full_results": full_records,
        "selected_primary": primary,
        "test_labels_used": False,
    }
    _json_dump(ARTIFACT_ROOT / "selection_results.json", result)
    print(f"selection winner: {primary['name']} "
          f"TSAUC={primary['validation_tsauc']:.6f}", flush=True)
    return result


def tune_joint() -> dict[str, object]:
    data = _load_data()
    selection_path = ARTIFACT_ROOT / "selection_results.json"
    selection = (json.loads(selection_path.read_text()) if selection_path.exists()
                 else run_selection())
    selected = selection["selected_primary"]
    columns, _, _ = _load_selection(selected["name"])
    records = []
    for name, overrides in TUNING_OVERRIDES:
        # Hold the selected model seed constant so this stage isolates the
        # declared hyperparameter change instead of turning seed luck into an
        # unreported tuning dimension.
        parameters = _parameters(MODEL_SEED + 100, overrides)
        job_tag = f"tune_fixedseed__{selected['name']}__{name}"
        job = _fit_validation_job(
            data, job_tag, columns,
            parameters, FULL_FRACTION, 900, 90,
        )
        records.append({
            "name": name, "overrides": overrides,
            "parameters": parameters,
            "validation_tsauc": job["validation_tsauc"],
            "best_iteration": job["best_iteration"],
            "job_tag": job_tag,
        })
    best = max(records, key=lambda row: row["validation_tsauc"])
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "selection_name": selected["name"],
        "selection_column_hash": _column_hash(columns),
        "selection_feature_count": int(columns.size),
        "configurations": records,
        "selected": best,
        "test_labels_used": False,
    }
    _json_dump(ARTIFACT_ROOT / "tuning_results.json", result)
    print(f"tuning winner: {best['name']} TSAUC={best['validation_tsauc']:.6f}",
          flush=True)
    return result


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(bounded) - np.log1p(-bounded)


def _logistic(margin: np.ndarray) -> np.ndarray:
    margin = np.asarray(margin, dtype=np.float64)
    result = np.empty_like(margin)
    positive = margin >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-margin[positive]))
    exponential = np.exp(margin[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def run_incumbent_baseline() -> dict[str, object]:
    """Retrain submission-3 feature contracts on this split and choose weight."""

    data = _load_data()
    core_columns = _prior_union(data)[:768]
    prior = _prior_union(data)
    candidate_columns = prior[768:]
    parameters = _parameters(MODEL_SEED + 700)
    core = _fit_validation_job(
        data, "baseline__core", core_columns, parameters,
        FULL_FRACTION, 700, 70,
    )
    candidate = _fit_validation_job(
        data, "baseline__candidate", candidate_columns, parameters,
        FULL_FRACTION, 700, 70,
    )
    core_prediction = np.load(core["validation_prediction_path"], allow_pickle=False)
    candidate_prediction = np.load(candidate["validation_prediction_path"],
                                   allow_pickle=False)
    valid_rows = np.flatnonzero(data.valid_full)
    target = data.metadata["target"][valid_rows]
    online_time = data.metadata["time_online"][valid_rows]
    core_margin = _safe_logit(core_prediction)
    candidate_margin = _safe_logit(candidate_prediction)
    search = []
    for core_weight in np.linspace(0.0, 1.0, 41):
        prediction = _logistic(
            core_weight * core_margin + (1.0 - core_weight) * candidate_margin
        )
        search.append({
            "core_weight": float(core_weight),
            "candidate_weight": float(1.0 - core_weight),
            "validation_tsauc": float(tsauc(target, prediction, online_time)),
        })
    best = max(search, key=lambda row: (
        row["validation_tsauc"], -abs(row["core_weight"] - 0.55)
    ))
    fixed_prediction = _logistic(0.55 * core_margin + 0.45 * candidate_margin)
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "baseline": "separate selected core/candidate LightGBMs",
        "parameters": parameters,
        "core_feature_count": int(core_columns.size),
        "candidate_feature_count": int(candidate_columns.size),
        "core_column_hash": _column_hash(core_columns),
        "candidate_column_hash": _column_hash(candidate_columns),
        "core_best_iteration": core["best_iteration"],
        "candidate_best_iteration": candidate["best_iteration"],
        "core_validation_tsauc": core["validation_tsauc"],
        "candidate_validation_tsauc": candidate["validation_tsauc"],
        "fixed_submission3_weight_validation_tsauc": float(
            tsauc(target, fixed_prediction, online_time)
        ),
        "weight_search": search,
        "selected": best,
        "test_labels_used": False,
    }
    _json_dump(ARTIFACT_ROOT / "baseline_validation.json", result)
    print(f"baseline: TSAUC={best['validation_tsauc']:.6f}, "
          f"core weight={best['core_weight']:.3f}", flush=True)
    return result


def freeze_development() -> dict[str, object]:
    """Write the immutable design contract before any test label is indexed."""

    frozen_path = ARTIFACT_ROOT / FROZEN_CONTRACT_FILE
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        expected = _sha_json({key: value for key, value in frozen.items()
                              if key != "contract_hash"})
        if frozen.get("contract_hash") != expected:
            raise ValueError("frozen contract hash is invalid")
        print("development contract already frozen", flush=True)
        return frozen
    data = _load_data()
    tuning_path = ARTIFACT_ROOT / "tuning_results.json"
    baseline_path = ARTIFACT_ROOT / "baseline_validation.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else tune_joint()
    baseline = (json.loads(baseline_path.read_text()) if baseline_path.exists()
                else run_incumbent_baseline())
    selection_name = tuning["selection_name"]
    columns, names, selection_metadata = _load_selection(selection_name)
    selected_path = ARTIFACT_ROOT / SELECTED_FEATURE_FILE
    np.savez_compressed(selected_path, columns=columns, names=np.asarray(names))
    offsets = data.offsets
    widths = data.widths
    family_counts = {
        family: int(np.sum((columns >= offset) & (columns < offset + width)))
        for family, offset, width in zip(FAMILY_NAMES, offsets, widths)
    }
    contract = {
        "experiment_version": EXPERIMENT_VERSION,
        "contract_revision": CONTRACT_REVISION,
        "created_without_test_labels": True,
        "split_contract_hash": _sha_json(json.loads(
            (ARTIFACT_ROOT / "split_contract.json").read_text()
        )),
        "selection": {
            "name": selection_name,
            "method_metadata": selection_metadata,
            "feature_count": int(columns.size),
            "column_hash": _column_hash(columns),
            "family_counts": family_counts,
            "selection_file": str(selected_path),
        },
        "joint_model": {
            "tuning_name": tuning["selected"]["name"],
            "validation_tsauc": tuning["selected"]["validation_tsauc"],
            "best_iteration": tuning["selected"]["best_iteration"],
            "parameters": tuning["selected"]["parameters"],
        },
        "incumbent": {
            "validation_tsauc": baseline["selected"]["validation_tsauc"],
            "core_weight": baseline["selected"]["core_weight"],
            "candidate_weight": baseline["selected"]["candidate_weight"],
            "core_best_iteration": baseline["core_best_iteration"],
            "candidate_best_iteration": baseline["candidate_best_iteration"],
            "parameters": baseline["parameters"],
            "core_column_hash": baseline["core_column_hash"],
            "candidate_column_hash": baseline["candidate_column_hash"],
        },
        "test_policy": (
            "Refit frozen models on train+validation; create all predictions "
            "before indexing test targets; score test exactly once."
        ),
        "contract_hash": "",
    }
    contract["contract_hash"] = _sha_json(
        {key: value for key, value in contract.items() if key != "contract_hash"}
    )
    _json_dump(frozen_path, contract)
    print(json.dumps(contract, indent=2), flush=True)
    return contract


def run_development() -> dict[str, object]:
    screen_rankings()
    run_selection()
    tune_joint()
    run_incumbent_baseline()
    return freeze_development()


def _fit_fixed(data: ExperimentData, name: str, columns: np.ndarray,
               parameters: dict[str, object], rounds: int) -> lgb.Booster:
    directory = ARTIFACT_ROOT / SEALED_MODEL_DIRECTORY / _safe(name)
    model_path = directory / "final.txt"
    manifest_path = directory / "manifest.json"
    contract = {
        "name": name, "column_hash": _column_hash(columns),
        "feature_count": int(columns.size), "parameters": parameters,
        "rounds": int(rounds), "sample_fraction": FULL_FRACTION,
        "sample_seed": SAMPLE_SEED, "training_roles": "train+validation",
    }
    contract_hash = _sha_json(contract)
    if model_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract_hash"] != contract_hash:
            raise ValueError(f"fixed model collision for {name}")
        return lgb.Booster(model_file=str(model_path))
    directory.mkdir(parents=True, exist_ok=True)
    sampled = _sample_mask(data, FULL_FRACTION)
    rows = np.flatnonzero(data.development_full & sampled)
    print(f"final {name}: {rows.size:,} development rows x "
          f"{columns.size:,} features, {rounds} rounds", flush=True)
    matrix = _materialize(data, rows, columns)
    target = data.metadata["target"][rows]
    weights = pair_weights_from_reference(
        target, data.metadata["time_online"][rows],
        data.metadata["target"][data.development_full],
        data.metadata["time_online"][data.development_full],
    )
    # Keep the frozen development seed as part of the model contract.  The
    # no-pretrained Crunch bundle consumes these exact parameters, so changing
    # seeds only for the sealed refit would evaluate a different learner than
    # the one that submission_5 can recreate from its train() iterator.
    final_parameters = dict(parameters)
    dataset = lgb.Dataset(
        matrix, label=target, weight=weights,
        feature_name=[data.names[int(index)] for index in columns],
        params=final_parameters, free_raw_data=True,
    )
    dataset.construct()
    del matrix, target, weights
    gc.collect()
    booster = lgb.train(final_parameters, dataset, num_boost_round=int(rounds))
    booster.save_model(model_path)
    _json_dump(manifest_path, {
        **contract, "contract_hash": contract_hash,
        "final_parameters": final_parameters, "training_rows": rows.size,
        "model_path": str(model_path),
    })
    del dataset
    gc.collect()
    return booster


def _subfold_groups(series_positions: np.ndarray, groups: int = 5) -> np.ndarray:
    unique = np.unique(series_positions)
    # Stable label-free integer mixing, then round-robin assignment for equal size.
    mixed = np.asarray([
        ((int(value) + 0x9E3779B9) * 0x85EBCA6B) & 0xFFFFFFFF
        for value in unique
    ], dtype=np.uint64)
    order = np.argsort(mixed, kind="stable")
    assignment = np.empty(unique.size, dtype=np.int8)
    assignment[order] = np.arange(unique.size) % groups
    return assignment[np.searchsorted(unique, series_positions)]


def _score_summary(target: np.ndarray, prediction: np.ndarray,
                   online_time: np.ndarray, series_position: np.ndarray) -> dict[str, object]:
    group = _subfold_groups(series_position)
    scores = np.asarray([
        tsauc(target[group == index], prediction[group == index],
              online_time[group == index])
        for index in range(5)
    ])
    standard_error = float(scores.std(ddof=1) / math.sqrt(scores.size))
    critical = 2.7764451051977987
    mean = float(scores.mean())
    return {
        "pooled_tsauc": float(tsauc(target, prediction, online_time)),
        "series_subfold_scores": scores.tolist(),
        "series_subfold_mean": mean,
        "series_subfold_standard_deviation": float(scores.std(ddof=1)),
        "series_subfold_standard_error": standard_error,
        "series_subfold_mean_95_percent_t_interval": [
            mean - critical * standard_error, mean + critical * standard_error
        ],
        "standard_error_note": (
            "Sample SD across five deterministic disjoint test-series groups "
            "divided by sqrt(5); rows are not treated as independent."
        ),
    }


def _cluster_bootstrap(target: np.ndarray, joint: np.ndarray, baseline: np.ndarray,
                       online_time: np.ndarray, series_position: np.ndarray,
                       replicates: int) -> dict[str, object]:
    unique, starts = np.unique(series_position, return_index=True)
    ends = np.r_[starts[1:], series_position.size]
    blocks = [np.arange(left, right) for left, right in zip(starts, ends)]
    rng = np.random.default_rng(20_260_829)
    joint_scores = np.empty(replicates)
    baseline_scores = np.empty(replicates)
    for replicate in range(replicates):
        sampled = rng.integers(0, unique.size, size=unique.size)
        rows = np.concatenate([blocks[int(index)] for index in sampled])
        joint_scores[replicate] = tsauc(
            target[rows], joint[rows], online_time[rows]
        )
        baseline_scores[replicate] = tsauc(
            target[rows], baseline[rows], online_time[rows]
        )
        if (replicate + 1) % 50 == 0:
            print(f"cluster bootstrap {replicate + 1}/{replicates}", flush=True)
    difference = joint_scores - baseline_scores

    def summary(values: np.ndarray) -> dict[str, object]:
        return {
            "standard_error": float(values.std(ddof=1)),
            "percentile_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
            "mean": float(values.mean()),
        }
    return {
        "replicates": replicates,
        "seed": 20_260_829,
        "sampling_unit": "complete test series with replacement",
        "joint": summary(joint_scores),
        "baseline": summary(baseline_scores),
        "joint_minus_baseline": summary(difference),
    }


def _write_result_summary(report: dict[str, object], frozen: dict[str, object]) -> None:
    joint = report["joint"]
    baseline = report["incumbent"]
    delta = report["paired_subfold_comparison"]
    bootstrap = report["series_cluster_bootstrap"]
    families = frozen["selection"]["family_counts"]
    pooled_difference = joint["pooled_tsauc"] - baseline["pooled_tsauc"]
    with np.load(frozen["selection"]["selection_file"], allow_pickle=False) as payload:
        novel_names = [
            str(name).removeprefix("d60__") for name in payload["names"]
            if str(name).startswith("d60__")
        ]
    novel_groups = {
        "robust scale/location": ("fvar", "levene", "abswelch", "fligner", "mood"),
        "expected-shortfall": ("cvar_",),
        "in-band/run": ("inband_",),
        "db4": ("db4_",),
        "distribution-distance": ("hellinger", "canberra", "braycurtis"),
        "Hurst": ("hurst_",),
    }
    novel_mix = {
        group: sum(any(name.startswith(prefix) for prefix in prefixes)
                   for name in novel_names)
        for group, prefixes in novel_groups.items()
    }
    text = f"""# Joint LightGBM experiment results

## Leakage contract

- Fresh series-grouped split seed: `{FOLD_SEED}`.
- Train/validation/test: 60%/20%/20% complete series.
- Feature ranking used train only; subset and parameters used validation only.
- The sealed test was scored once after `{frozen['contract_hash']}` was frozen.
- The detector bank is fixed and every feature has offline/streaming parity.
- Historical experimentation means this is a procedurally fresh holdout, not a
  claim that these training series were never seen during earlier research.

## Selected joint model

- Selection: `{frozen['selection']['name']}`
- Features: `{frozen['selection']['feature_count']}` total: `{families['core']}`
  core, `{families['candidate']}` candidate, and `{families['novel']}` from the
  new winner-inspired bank.
- Selected new-feature mix: `{novel_mix['robust scale/location']}` robust
  scale/location, `{novel_mix['expected-shortfall']}` expected-shortfall,
  `{novel_mix['in-band/run']}` in-band/run, `{novel_mix['db4']}` db4,
  `{novel_mix['distribution-distance']}` distribution-distance, and
  `{novel_mix['Hurst']}` Hurst columns.
- LightGBM tuning configuration: `{frozen['joint_model']['tuning_name']}`
- Validation TSAUC: `{frozen['joint_model']['validation_tsauc']:.6f}`
- Sealed-test TSAUC: `{joint['pooled_tsauc']:.6f}`
- Five-series-group mean ± SE: `{joint['series_subfold_mean']:.6f} ± {joint['series_subfold_standard_error']:.6f}`
- Complete-series bootstrap SE: `{bootstrap['joint']['standard_error']:.6f}`
- Bootstrap 95% interval: `[{bootstrap['joint']['percentile_95_interval'][0]:.6f}, {bootstrap['joint']['percentile_95_interval'][1]:.6f}]`

## Same-split incumbent

- Separate selected core/candidate LightGBMs.
- Validation TSAUC: `{frozen['incumbent']['validation_tsauc']:.6f}`
- Sealed-test TSAUC: `{baseline['pooled_tsauc']:.6f}`
- Five-series-group mean ± SE: `{baseline['series_subfold_mean']:.6f} ± {baseline['series_subfold_standard_error']:.6f}`
- Joint minus incumbent pooled TSAUC: `{pooled_difference:+.6f}`
- Joint minus incumbent subfold mean: `{delta['mean_difference']:+.6f}`
- Paired SE: `{delta['standard_error']:.6f}`
- Paired 95% t interval: `[{delta['95_percent_t_interval'][0]:+.6f}, {delta['95_percent_t_interval'][1]:+.6f}]`
- Paired complete-series bootstrap SE: `{bootstrap['joint_minus_baseline']['standard_error']:.6f}`
- Paired bootstrap 95% interval: `[{bootstrap['joint_minus_baseline']['percentile_95_interval'][0]:+.6f}, {bootstrap['joint_minus_baseline']['percentile_95_interval'][1]:+.6f}]`

## Submission decision

`submission_5` eligibility: **{'PASS' if report['submission_5_eligible'] else 'FAIL'}**.
The folder is built only when the frozen joint model has strictly higher pooled
sealed-test TSAUC than the same-split incumbent. No post-test retuning is allowed.
"""
    (PROJECT / "models" / "joint_lightgbm_results.md").write_text(
        text, encoding="utf-8"
    )


def evaluate_test(bootstrap_replicates: int = 200) -> dict[str, object]:
    """Refit frozen models, generate predictions, then open test labels once."""

    report_path = ARTIFACT_ROOT / SEALED_RESULT_FILE
    if report_path.exists():
        print(report_path.read_text(), flush=True)
        return json.loads(report_path.read_text())
    frozen_path = ARTIFACT_ROOT / FROZEN_CONTRACT_FILE
    if not frozen_path.exists():
        raise RuntimeError("run and freeze development before evaluating test")
    frozen = json.loads(frozen_path.read_text())
    expected = _sha_json({key: value for key, value in frozen.items()
                          if key != "contract_hash"})
    if frozen["contract_hash"] != expected:
        raise ValueError("frozen contract has been modified")
    data = _load_data()
    with np.load(frozen["selection"]["selection_file"], allow_pickle=False) as payload:
        joint_columns = payload["columns"].astype(np.int32)
    prior = _prior_union(data)
    core_columns, candidate_columns = prior[:768], prior[768:]
    joint_booster = _fit_fixed(
        data, "joint", joint_columns, frozen["joint_model"]["parameters"],
        frozen["joint_model"]["best_iteration"],
    )
    core_booster = _fit_fixed(
        data, "incumbent_core", core_columns,
        frozen["incumbent"]["parameters"],
        frozen["incumbent"]["core_best_iteration"],
    )
    candidate_booster = _fit_fixed(
        data, "incumbent_candidate", candidate_columns,
        frozen["incumbent"]["parameters"],
        frozen["incumbent"]["candidate_best_iteration"],
    )
    test_rows = np.flatnonzero(data.test_full)
    prediction_directory = ARTIFACT_ROOT / SEALED_PREDICTION_DIRECTORY
    prediction_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "joint": prediction_directory / "joint.f32.npy",
        "core": prediction_directory / "incumbent_core.f32.npy",
        "candidate": prediction_directory / "incumbent_candidate.f32.npy",
    }
    boosters = {"joint": (joint_booster, joint_columns),
                "core": (core_booster, core_columns),
                "candidate": (candidate_booster, candidate_columns)}
    predictions = {}
    for name, (booster, columns) in boosters.items():
        if paths[name].exists():
            predictions[name] = np.load(paths[name], allow_pickle=False)
        else:
            print(f"sealed test: predicting {name} without reading labels", flush=True)
            predictions[name] = _predict(booster, data, test_rows, columns)
            np.save(paths[name], predictions[name])
    core_weight = float(frozen["incumbent"]["core_weight"])
    baseline_prediction = _logistic(
        core_weight * _safe_logit(predictions["core"])
        + (1.0 - core_weight) * _safe_logit(predictions["candidate"])
    )

    # Everything above is label-free. This sentinel is committed immediately
    # before the only test-target indexing operation in the workflow.
    sentinel_path = ARTIFACT_ROOT / SEALED_SENTINEL_FILE
    if sentinel_path.exists():
        raise RuntimeError(
            "sealed-test sentinel exists without a completed report; inspect "
            "the interrupted run before explicitly removing the sentinel"
        )
    _json_dump(sentinel_path, {
        "frozen_contract_hash": frozen["contract_hash"],
        "prediction_hashes": {
            name: hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()
            for name, value in predictions.items()
        },
        "purpose": "one-shot scoring; no further model changes permitted",
    })
    target = data.metadata["target"][test_rows]
    online_time = data.metadata["time_online"][test_rows]
    series_position = data.metadata["series_position"][test_rows]
    joint_summary = _score_summary(
        target, predictions["joint"], online_time, series_position
    )
    baseline_summary = _score_summary(
        target, baseline_prediction, online_time, series_position
    )
    group = _subfold_groups(series_position)
    differences = np.asarray([
        tsauc(target[group == index], predictions["joint"][group == index],
              online_time[group == index])
        - tsauc(target[group == index], baseline_prediction[group == index],
                online_time[group == index])
        for index in range(5)
    ])
    difference_mean = float(differences.mean())
    difference_se = float(differences.std(ddof=1) / math.sqrt(5))
    critical = 2.7764451051977987
    report = {
        "experiment_version": EXPERIMENT_VERSION,
        "frozen_contract_hash": frozen["contract_hash"],
        "test_opened_after_freeze": True,
        "test_series": int(np.unique(series_position).size),
        "test_rows": int(test_rows.size),
        "joint": joint_summary,
        "incumbent": baseline_summary,
        "paired_subfold_comparison": {
            "joint_minus_incumbent_by_subfold": differences.tolist(),
            "mean_difference": difference_mean,
            "standard_error": difference_se,
            "95_percent_t_interval": [
                difference_mean - critical * difference_se,
                difference_mean + critical * difference_se,
            ],
        },
        "series_cluster_bootstrap": _cluster_bootstrap(
            target, predictions["joint"], baseline_prediction,
            online_time, series_position, bootstrap_replicates,
        ),
        "submission_5_eligible": bool(
            joint_summary["pooled_tsauc"] > baseline_summary["pooled_tsauc"]
        ),
        "decision_rule": (
            "Build submission_5 only when frozen joint pooled test TSAUC is "
            "strictly greater than the same-split incumbent; never retune."
        ),
    }
    _json_dump(report_path, report)
    _write_result_summary(report, frozen)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("screen")
    subparsers.add_parser("select")
    subparsers.add_parser("tune")
    subparsers.add_parser("baseline")
    subparsers.add_parser("freeze")
    subparsers.add_parser("development")
    evaluate = subparsers.add_parser("evaluate-test")
    evaluate.add_argument("--bootstrap-replicates", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "screen":
        screen_rankings()
    elif args.command == "select":
        run_selection()
    elif args.command == "tune":
        tune_joint()
    elif args.command == "baseline":
        run_incumbent_baseline()
    elif args.command == "freeze":
        freeze_development()
    elif args.command == "development":
        run_development()
    elif args.command == "evaluate-test":
        if args.bootstrap_replicates < 20:
            raise ValueError("use at least 20 cluster-bootstrap replicates")
        evaluate_test(args.bootstrap_replicates)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
