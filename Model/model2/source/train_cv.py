"""Build features, train the fold ensemble, and report honest TS-AUC.

Typical usage from the repository root::

    python3 train_cv.py build-cache
    python3 train_cv.py cross-validate --model-directory artifacts/model_core

The folds split entire series, never rows.  Each fold model is trained on a
deterministic subsample of online decisions from the other four folds and is
then evaluated on *every* online decision in its held-out series.  The five
models are also the final bagged inference ensemble: at test time all five
vote, while the out-of-fold score for a training row comes only from a model
that never saw any row from that series.
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
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from competition_data import TrainingData
from feature_pipeline import (
    ALL_BANKS,
    CORE_BANKS,
    bank_slices,
    build_feature_cache,
    feature_names,
    load_feature_cache,
)
from tsauc import fold_standard_error, tsauc, tsauc_details
from postprocess import DEFAULT_PENALTIES, causal_decayed_max


DEFAULT_MODEL_PARAMETERS: dict[str, object] = {
    "objective": "binary",
    # Early stopping is driven by the custom exact TS-AUC evaluator below.
    # Ordinary row-level log loss/AUC rewards cross-time comparisons that the
    # competition never makes and stopped materially too early in smoke runs.
    "metric": "None",
    "boosting_type": "gbdt",
    "learning_rate": 0.035,
    "num_leaves": 31,
    "min_data_in_leaf": 300,
    "max_bin": 127,
    "feature_fraction": 0.60,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.20,
    "lambda_l2": 3.0,
    "min_gain_to_split": 0.0,
    "verbosity": -1,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
    "deterministic": True,
    "force_col_wise": True,
}


@dataclass
class HorizontalFeatureMatrix:
    """A row-aligned virtual horizontal concatenation of cache memmaps."""

    matrices: list[np.memmap]

    def __post_init__(self) -> None:
        if not self.matrices:
            raise ValueError("at least one feature matrix is required")
        rows = {matrix.shape[0] for matrix in self.matrices}
        if len(rows) != 1:
            raise ValueError("feature caches have different row counts")
        self.shape = (
            self.matrices[0].shape[0],
            sum(matrix.shape[1] for matrix in self.matrices),
        )

    def __getitem__(self, item):
        if len(self.matrices) == 1:
            return self.matrices[0][item]
        return np.concatenate(
            [np.asarray(matrix[item], dtype=np.float32) for matrix in self.matrices],
            axis=1,
        )


def load_aligned_caches(
    cache_directory: str | Path,
    cache_names: Sequence[str],
):
    """Load one or more row-aligned caches and verify their metadata."""

    caches = [load_feature_cache(cache_directory, name) for name in cache_names]
    reference = caches[0].load_metadata()
    for cache in caches[1:]:
        candidate = cache.load_metadata()
        for key in ("row_id", "series_position", "time_online", "target"):
            if not np.array_equal(reference[key], candidate[key]):
                raise ValueError(f"cache {cache.matrix_path} is not row-aligned ({key})")
    matrix = HorizontalFeatureMatrix([cache.open("r") for cache in caches])
    banks = tuple(bank for cache in caches for bank in cache.banks)
    if len(set(banks)) != len(banks):
        raise ValueError("feature caches contain duplicate banks")
    return caches, reference, matrix, banks


def make_tsauc_evaluator(time_online: np.ndarray):
    """Create a LightGBM callback metric for one fixed validation row order."""

    step = np.asarray(time_online)

    def evaluate(prediction: np.ndarray, dataset: lgb.Dataset):
        labels = dataset.get_label().astype(np.uint8, copy=False)
        return "tsauc", tsauc(labels, prediction, step), True

    return evaluate


def make_series_folds(
    online_lengths: np.ndarray,
    tau_index: np.ndarray,
    n_splits: int = 5,
    seed: int = 73051,
) -> np.ndarray:
    """Return a stratified fold number for each series.

    Stratification balances three important sources of TS-AUC difficulty:
    break/no-break, online horizon, and (for broken series) relative break
    location.  It uses no time-series values or detector outputs.
    """

    lengths = np.asarray(online_lengths, dtype=np.int64)
    tau = np.asarray(tau_index, dtype=np.int64)
    if lengths.shape != tau.shape:
        raise ValueError("online_lengths and tau_index shapes differ")
    length_bin = np.asarray(
        pd.qcut(lengths, 5, labels=False, duplicates="drop"), dtype=np.int64
    )
    has_break = tau >= 0
    relative_bin = np.full(tau.size, -1, dtype=np.int64)
    relative_bin[has_break] = np.minimum(
        4,
        np.floor(5.0 * tau[has_break] / lengths[has_break]).astype(np.int64),
    )
    strata = np.asarray(
        [
            f"{int(broken)}-{int(horizon)}-{int(location)}"
            for broken, horizon, location in zip(
                has_break, length_bin, relative_bin
            )
        ]
    )
    folds = np.empty(tau.size, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(tau.size), strata)):
        folds[held] = fold
    return folds


def deterministic_sample_mask(
    row_id: np.ndarray,
    time_online: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Stable uniform row sample independent of labels and Python hashing."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample fraction must be in (0, 1]")
    if fraction == 1.0:
        return np.ones(len(row_id), dtype=bool)
    # SplitMix64-style mixing gives each (series, time) a reproducible draw.
    value = (
        np.asarray(row_id, dtype=np.uint64) * np.uint64(0x9E3779B185EBCA87)
        + np.asarray(time_online, dtype=np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        + np.uint64(seed)
    )
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    threshold = int(fraction * 1_000_003)
    return (value % np.uint64(1_000_003)) < threshold


def pair_weights_from_reference(
    sample_target: np.ndarray,
    sample_time: np.ndarray,
    reference_target: np.ndarray,
    reference_time: np.ndarray,
) -> np.ndarray:
    """TS-AUC-aligned row weights using full-fold class counts by time."""

    maximum = int(max(sample_time.max(initial=0), reference_time.max(initial=0)))
    count = np.bincount(reference_time, minlength=maximum + 1)
    positive = np.bincount(
        reference_time, weights=reference_target, minlength=maximum + 1
    )
    negative = count - positive
    weights = np.where(
        sample_target == 1, negative[sample_time], positive[sample_time]
    ).astype(np.float64)
    useful = weights > 0
    if np.any(useful):
        weights[useful] /= weights[useful].mean()
    return weights.astype(np.float32)


def selected_columns(
    cached_banks: Sequence[str], use_banks: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Resolve a bank subset to physical cache columns and names."""

    slices = bank_slices(cached_banks)
    unknown = set(use_banks) - set(slices)
    if unknown:
        raise ValueError(f"requested banks absent from cache: {sorted(unknown)}")
    columns = np.concatenate(
        [np.arange(slices[bank].start, slices[bank].stop) for bank in use_banks]
    ).astype(np.int32)
    names_by_bank = {
        bank: feature_names(cached_banks)[slices[bank]] for bank in cached_banks
    }
    names = [name for bank in use_banks for name in names_by_bank[bank]]
    return columns, names


def matrix_rows(
    matrix,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Materialize selected cache rows/columns as contiguous float32."""

    # Reading rows first is substantially faster for the row-major memmap.
    block = np.asarray(matrix[rows], dtype=np.float32)
    if columns.size != matrix.shape[1] or not np.array_equal(
        columns, np.arange(matrix.shape[1])
    ):
        block = block[:, columns]
    return np.ascontiguousarray(block, dtype=np.float32)


def predict_rows(
    booster: lgb.Booster,
    matrix,
    rows: np.ndarray,
    columns: np.ndarray,
    chunk_rows: int,
) -> np.ndarray:
    """Predict arbitrary sorted rows without materializing a whole fold."""

    output = np.empty(rows.size, dtype=np.float32)
    iteration = booster.best_iteration or booster.current_iteration()
    for left in range(0, rows.size, chunk_rows):
        right = min(left + chunk_rows, rows.size)
        block = matrix_rows(matrix, rows[left:right], columns)
        output[left:right] = booster.predict(
            block, num_iteration=iteration
        ).astype(np.float32)
    return output


def _configuration_hash(configuration: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()


def cross_validate(args: argparse.Namespace) -> dict[str, object]:
    cache_names = (args.cache_name, *args.additional_cache_name)
    caches, metadata, matrix, cached_banks = load_aligned_caches(
        args.cache_directory, cache_names
    )
    use_banks = tuple(args.banks or cached_banks)
    columns, names = selected_columns(cached_banks, use_banks)

    series_fold = make_series_folds(
        metadata["online_lengths"],
        metadata["tau_index"],
        n_splits=args.folds,
        seed=args.fold_seed,
    )
    row_fold = series_fold[metadata["series_position"]]
    sampled = deterministic_sample_mask(
        metadata["row_id"],
        metadata["time_online"],
        args.sample_fraction,
        args.sample_seed,
    )
    eligible = metadata["time_online"] >= args.time_min
    if args.time_max is not None:
        eligible &= metadata["time_online"] < args.time_max
    if not np.any(eligible):
        raise ValueError("selected time range contains no online rows")

    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    parameters["num_threads"] = args.threads
    for argument, parameter in (
        ("learning_rate", "learning_rate"),
        ("num_leaves", "num_leaves"),
        ("min_data_in_leaf", "min_data_in_leaf"),
        ("feature_fraction", "feature_fraction"),
    ):
        value = getattr(args, argument)
        if value is not None:
            parameters[parameter] = value
    cache_signatures = [
        json.loads(cache.manifest_path.read_text())["feature_signature"]
        for cache in caches
    ]
    configuration: dict[str, object] = {
        "banks": use_banks,
        "feature_names": names,
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "sample_fraction": args.sample_fraction,
        "sample_seed": args.sample_seed,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "parameters": parameters,
    }
    # Preserve the original single-cache schema so a partially completed CV
    # run remains exactly resumable after multi-cache support was added.
    if len(caches) == 1:
        configuration["cache_feature_signature"] = cache_signatures[0]
    else:
        configuration["cache_feature_signatures"] = cache_signatures
        configuration["cache_names"] = cache_names
    # Omit the all-time defaults to retain exact compatibility with model
    # manifests created before causal time-regime support was introduced.
    if args.time_min != 0:
        configuration["time_min"] = args.time_min
    if args.time_max is not None:
        configuration["time_max"] = args.time_max
    config_hash = _configuration_hash(configuration)

    model_directory = Path(args.model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = model_directory / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get("configuration_hash") != config_hash:
            raise ValueError(
                "model directory contains an incompatible run; choose another "
                "--model-directory or restore its original arguments"
            )
    else:
        manifest_path.write_text(
            json.dumps(
                {
                    **configuration,
                    "configuration_hash": config_hash,
                    **(
                        {"cache_name": cache_names[0]}
                        if len(cache_names) == 1
                        else {"cache_names": cache_names}
                    ),
                },
                indent=2,
            )
        )

    oof_path = model_directory / "oof_prediction.f32.mmap"
    if oof_path.exists():
        if oof_path.stat().st_size != matrix.shape[0] * 4:
            raise ValueError("OOF file has an unexpected size")
        oof = np.memmap(
            oof_path, dtype=np.float32, mode="r+", shape=(matrix.shape[0],)
        )
    else:
        oof = np.memmap(
            oof_path, dtype=np.float32, mode="w+", shape=(matrix.shape[0],)
        )
        oof[:] = np.nan
        oof.flush()

    progress_path = model_directory / "cv_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    else:
        progress = {"configuration_hash": config_hash, "folds": {}}
    if progress.get("configuration_hash") != config_hash:
        raise ValueError("CV progress belongs to a different configuration")

    requested_folds = (
        [args.only_fold] if args.only_fold is not None else list(range(args.folds))
    )
    for fold in requested_folds:
        if not 0 <= fold < args.folds:
            raise ValueError(f"invalid fold {fold}")
        model_path = model_directory / f"fold_{fold}.txt"
        held_rows = np.flatnonzero((row_fold == fold) & eligible)
        fold_key = str(fold)
        already_complete = (
            fold_key in progress["folds"]
            and model_path.exists()
            and np.all(np.isfinite(oof[held_rows]))
        )
        if already_complete:
            print(f"fold {fold}: already complete", flush=True)
            continue

        train_full = (row_fold != fold) & eligible
        train_rows = np.flatnonzero(train_full & sampled)
        valid_rows = np.flatnonzero((row_fold == fold) & eligible & sampled)
        print(
            f"fold {fold}: materializing {train_rows.size:,} train and "
            f"{valid_rows.size:,} early-stop rows, {columns.size:,} features",
            flush=True,
        )
        started = time.perf_counter()
        x_train = matrix_rows(matrix, train_rows, columns)
        x_valid = matrix_rows(matrix, valid_rows, columns)
        y_train = metadata["target"][train_rows]
        y_valid = metadata["target"][valid_rows]
        weights_train = pair_weights_from_reference(
            y_train,
            metadata["time_online"][train_rows],
            metadata["target"][train_full],
            metadata["time_online"][train_full],
        )
        held_full = (row_fold == fold) & eligible
        weights_valid = pair_weights_from_reference(
            y_valid,
            metadata["time_online"][valid_rows],
            metadata["target"][held_full],
            metadata["time_online"][held_full],
        )
        fold_parameters = dict(parameters)
        fold_parameters.update(
            {
                "seed": args.model_seed + fold,
                "feature_fraction_seed": args.model_seed + 100 + fold,
                "bagging_seed": args.model_seed + 200 + fold,
                "data_random_seed": args.model_seed + 300 + fold,
            }
        )
        train_set = lgb.Dataset(
            x_train,
            label=y_train,
            weight=weights_train,
            feature_name=names,
            params=fold_parameters,
            free_raw_data=True,
        )
        valid_set = lgb.Dataset(
            x_valid,
            label=y_valid,
            weight=weights_valid,
            feature_name=names,
            reference=train_set,
            params=fold_parameters,
            free_raw_data=True,
        )
        train_set.construct()
        valid_set.construct()
        del x_train, x_valid, y_train, y_valid, weights_train, weights_valid
        gc.collect()

        booster = lgb.train(
            fold_parameters,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[valid_set],
            valid_names=["held_sample"],
            feval=make_tsauc_evaluator(metadata["time_online"][valid_rows]),
            callbacks=[
                lgb.early_stopping(
                    args.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=True,
                ),
                lgb.log_evaluation(period=args.log_every),
            ],
        )
        booster.save_model(model_path, num_iteration=booster.best_iteration)
        del train_set, valid_set
        gc.collect()

        print(
            f"fold {fold}: predicting all {held_rows.size:,} held-out rows",
            flush=True,
        )
        held_prediction = predict_rows(
            booster,
            matrix,
            held_rows,
            columns,
            args.prediction_chunk_rows,
        )
        oof[held_rows] = held_prediction
        oof.flush()
        fold_score = tsauc(
            metadata["target"][held_rows],
            held_prediction,
            metadata["time_online"][held_rows],
        )
        elapsed = time.perf_counter() - started
        progress["folds"][fold_key] = {
            "score": fold_score,
            "best_iteration": int(booster.best_iteration),
            "held_series": int(np.sum(series_fold == fold)),
            "held_rows": int(held_rows.size),
            "train_sample_rows": int(train_rows.size),
            "elapsed_seconds": elapsed,
        }
        progress_path.write_text(json.dumps(progress, indent=2))
        print(
            f"fold {fold}: TS-AUC={fold_score:.6f}, "
            f"iteration={booster.best_iteration}, elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
        del booster, held_prediction
        gc.collect()

    completed = sorted(int(key) for key in progress["folds"])
    results: dict[str, object] = {
        "configuration_hash": config_hash,
        "completed_folds": completed,
        "folds": progress["folds"],
    }
    if len(completed) == args.folds and np.all(np.isfinite(oof[eligible])):
        fold_scores = np.asarray(
            [progress["folds"][str(fold)]["score"] for fold in range(args.folds)]
        )
        details = tsauc_details(
            metadata["target"][eligible],
            oof[eligible],
            metadata["time_online"][eligible],
        )
        standard_error = fold_standard_error(fold_scores)
        # With five folds the 95% t critical value is 2.776.  Use scipy only
        # when the number of folds is changed.
        if args.folds == 5:
            critical = 2.7764451051977987
        else:
            from scipy.stats import t as student_t

            critical = float(student_t.ppf(0.975, df=args.folds - 1))
        mean_score = float(fold_scores.mean())
        results.update(
            {
                "pooled_oof_tsauc": details.score,
                "fold_mean_tsauc": mean_score,
                "fold_standard_deviation": float(fold_scores.std(ddof=1)),
                "fold_standard_error": standard_error,
                "fold_mean_95_percent_ci": [
                    mean_score - critical * standard_error,
                    mean_score + critical * standard_error,
                ],
                "scored_time_steps": details.scored_time_steps,
                "total_pair_weight": details.total_pair_weight,
            }
        )

        # Diagnostics for the absorbing target.  These values are reported,
        # not silently selected: raw OOF remains the primary unbiased score.
        # A deployment penalty is adopted only if its gain is consistent in
        # every fold (and that choice is documented in the final manifest).
        if args.time_min == 0 and args.time_max is None:
            postprocess_grid: dict[str, object] = {}
            for penalty in DEFAULT_PENALTIES:
                transformed = causal_decayed_max(
                    oof, metadata["series_position"], penalty
                )
                scores = [
                    tsauc(
                        metadata["target"][row_fold == fold],
                        transformed[row_fold == fold],
                        metadata["time_online"][row_fold == fold],
                    )
                    for fold in range(args.folds)
                ]
                name = "raw" if penalty is None else f"decayed_max_{penalty:g}"
                postprocess_grid[name] = {
                    "pooled_tsauc": tsauc(
                        metadata["target"], transformed, metadata["time_online"]
                    ),
                    "fold_scores": scores,
                    "fold_mean": float(np.mean(scores)),
                    "fold_standard_error": fold_standard_error(scores),
                }
            results["postprocess_grid"] = postprocess_grid

        # Aggregate split/gain importance across the final fold ensemble.
        importance = pd.DataFrame({"feature": names})
        for fold in range(args.folds):
            booster = lgb.Booster(model_file=str(model_directory / f"fold_{fold}.txt"))
            importance[f"gain_fold_{fold}"] = booster.feature_importance(
                importance_type="gain"
            )
            importance[f"split_fold_{fold}"] = booster.feature_importance(
                importance_type="split"
            )
        gain_columns = [f"gain_fold_{fold}" for fold in range(args.folds)]
        split_columns = [f"split_fold_{fold}" for fold in range(args.folds)]
        importance["gain_mean"] = importance[gain_columns].mean(axis=1)
        importance["split_mean"] = importance[split_columns].mean(axis=1)
        importance.sort_values("gain_mean", ascending=False).to_csv(
            model_directory / "feature_importance.csv", index=False
        )

    results_path = model_directory / "cv_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    return results


def fit_final(args: argparse.Namespace) -> Path:
    """Fit one deployment model on all sampled training series.

    Cross-validation models deliberately omit one fold apiece.  Once their
    score has fixed the design and iteration count, this final fit recovers
    the missing 20% of labeled series and is much cheaper to call once per
    streaming observation than a five-model ensemble.
    """

    model_directory = Path(args.model_directory)
    manifest_path = model_directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"run cross-validate first; missing {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    cache_names = tuple(manifest.get("cache_names", [args.cache_name]))
    caches, metadata, matrix, cached_banks = load_aligned_caches(
        args.cache_directory, cache_names
    )
    use_banks = tuple(manifest["banks"])
    columns, names = selected_columns(cached_banks, use_banks)
    if names != manifest["feature_names"]:
        raise ValueError("feature names no longer match the CV manifest")

    if args.iterations is not None:
        iterations = args.iterations
    else:
        results_path = model_directory / "cv_results.json"
        results = json.loads(results_path.read_text())
        completed = results.get("completed_folds", [])
        if len(completed) != int(manifest["folds"]):
            raise ValueError("all CV folds must finish before automatic final fit")
        iterations = int(
            round(
                np.median(
                    [
                        results["folds"][str(fold)]["best_iteration"]
                        for fold in completed
                    ]
                )
            )
        )

    parameters = dict(manifest["parameters"])
    parameters["num_threads"] = args.threads
    parameters.update(
        {
            "seed": args.model_seed,
            "feature_fraction_seed": args.model_seed + 100,
            "bagging_seed": args.model_seed + 200,
            "data_random_seed": args.model_seed + 300,
        }
    )

    sampled = deterministic_sample_mask(
        metadata["row_id"],
        metadata["time_online"],
        float(manifest["sample_fraction"]),
        int(manifest["sample_seed"]),
    )
    eligible = metadata["time_online"] >= int(manifest.get("time_min", 0))
    time_max = manifest.get("time_max")
    if time_max is not None:
        eligible &= metadata["time_online"] < int(time_max)
    rows = np.flatnonzero(sampled & eligible)
    print(
        f"final: materializing {rows.size:,} rows and {columns.size:,} features",
        flush=True,
    )
    x_train = matrix_rows(matrix, rows, columns)
    y_train = metadata["target"][rows]
    weights = pair_weights_from_reference(
        y_train,
        metadata["time_online"][rows],
        metadata["target"],
        metadata["time_online"],
    )
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        feature_name=names,
        params=parameters,
        free_raw_data=True,
    )
    dataset.construct()
    del x_train, y_train, weights
    gc.collect()

    print(f"final: training {iterations} boosting rounds", flush=True)
    booster = lgb.train(parameters, dataset, num_boost_round=iterations)
    output_path = model_directory / "final.txt"
    booster.save_model(output_path)
    manifest["final_model"] = {
        "path": output_path.name,
        "iterations": iterations,
        "training_rows": int(rows.size),
        "model_seed": args.model_seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"final: saved {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("build-cache")
    cache_parser.add_argument("--train-directory", default="train")
    cache_parser.add_argument("--cache-directory", default="artifacts")
    cache_parser.add_argument("--cache-name", default="core_features")
    cache_parser.add_argument("--progress-every", type=int, default=25)
    cache_parser.add_argument("--workers", type=int, default=1)
    cache_parser.add_argument("--banks", nargs="+", choices=ALL_BANKS)

    cv_parser = subparsers.add_parser("cross-validate")
    cv_parser.add_argument("--cache-directory", default="artifacts")
    cv_parser.add_argument("--cache-name", default="core_features")
    cv_parser.add_argument("--additional-cache-name", nargs="*", default=[])
    cv_parser.add_argument("--model-directory", default="artifacts/model_core")
    cv_parser.add_argument("--banks", nargs="+", choices=ALL_BANKS)
    cv_parser.add_argument("--folds", type=int, default=5)
    cv_parser.add_argument("--fold-seed", type=int, default=73051)
    cv_parser.add_argument("--sample-fraction", type=float, default=0.20)
    cv_parser.add_argument("--sample-seed", type=int, default=99173)
    cv_parser.add_argument("--model-seed", type=int, default=41413)
    cv_parser.add_argument("--num-boost-round", type=int, default=900)
    cv_parser.add_argument("--early-stopping-rounds", type=int, default=100)
    cv_parser.add_argument("--prediction-chunk-rows", type=int, default=50_000)
    cv_parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    cv_parser.add_argument("--learning-rate", type=float)
    cv_parser.add_argument("--num-leaves", type=int)
    cv_parser.add_argument("--min-data-in-leaf", type=int)
    cv_parser.add_argument("--feature-fraction", type=float)
    cv_parser.add_argument("--log-every", type=int, default=50)
    cv_parser.add_argument("--only-fold", type=int)
    cv_parser.add_argument("--time-min", type=int, default=0)
    cv_parser.add_argument("--time-max", type=int)

    final_parser = subparsers.add_parser("fit-final")
    final_parser.add_argument("--cache-directory", default="artifacts")
    final_parser.add_argument("--cache-name", default="core_features")
    final_parser.add_argument("--model-directory", default="artifacts/model_core")
    final_parser.add_argument("--iterations", type=int)
    final_parser.add_argument("--model-seed", type=int, default=51413)
    final_parser.add_argument(
        "--threads", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-cache":
        data = TrainingData(args.train_directory)
        cache = build_feature_cache(
            data,
            cache_directory=args.cache_directory,
            name=args.cache_name,
            banks=tuple(args.banks or CORE_BANKS),
            progress_every=args.progress_every,
            workers=args.workers,
        )
        print(
            f"ready: {cache.matrix_path} ({cache.rows:,} x {cache.columns:,})"
        )
    elif args.command == "cross-validate":
        cross_validate(args)
    elif args.command == "fit-final":
        fit_final(args)
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
