"""Nested series-fold stacking for the core and candidate OOF predictions.

Base predictions for every row are already out of fold.  A stacker scoring
fold k is trained only on base OOF predictions and labels from folds != k.
This preserves the full-series leakage barrier while allowing a small model
to learn time-dependent weighting and disagreement patterns.  The final
stacker is fit on all OOF rows only after nested performance is reported.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

from feature_pipeline import load_feature_cache
from train_cv import (
    deterministic_sample_mask,
    make_series_folds,
    make_tsauc_evaluator,
    pair_weights_from_reference,
)
from tsauc import fold_standard_error, tsauc


STACK_FEATURE_NAMES = (
    "core_margin",
    "candidate_margin",
    "mean_margin",
    "margin_difference",
    "absolute_margin_difference",
    "maximum_margin",
    "minimum_margin",
    "core_probability",
    "candidate_probability",
    "log_time",
    "core_margin_x_log_time",
    "candidate_margin_x_log_time",
    "difference_x_log_time",
)

STACK_PARAMETERS: dict[str, object] = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 1_000,
    "max_bin": 127,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def safe_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def stack_features(
    core_probability: np.ndarray,
    candidate_probability: np.ndarray,
    time_online: np.ndarray,
) -> np.ndarray:
    """Materialize the documented, causal two-model stacking contract."""

    core = np.asarray(core_probability, dtype=np.float64)
    candidate = np.asarray(candidate_probability, dtype=np.float64)
    step = np.asarray(time_online, dtype=np.float64)
    if core.shape != candidate.shape or core.shape != step.shape:
        raise ValueError("stacking inputs have different shapes")
    core_margin = safe_logit(core)
    candidate_margin = safe_logit(candidate)
    difference = core_margin - candidate_margin
    log_time = np.log1p(step) / np.log(1_000.0)
    output = np.column_stack(
        (
            core_margin,
            candidate_margin,
            0.5 * (core_margin + candidate_margin),
            difference,
            np.abs(difference),
            np.maximum(core_margin, candidate_margin),
            np.minimum(core_margin, candidate_margin),
            core,
            candidate,
            log_time,
            core_margin * log_time,
            candidate_margin * log_time,
            difference * log_time,
        )
    )
    return np.ascontiguousarray(output, dtype=np.float32)


def _load_inputs(args: argparse.Namespace):
    metadata = load_feature_cache(
        args.cache_directory, args.cache_name
    ).load_metadata()
    count = metadata["target"].size
    predictions = []
    for directory in (args.core_directory, args.candidate_directory):
        values = np.memmap(
            Path(directory) / "oof_prediction.f32.mmap",
            dtype=np.float32,
            mode="r",
            shape=(count,),
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"base OOF predictions are incomplete: {directory}")
        predictions.append(values)
    series_fold = make_series_folds(
        metadata["online_lengths"],
        metadata["tau_index"],
        n_splits=args.folds,
        seed=args.fold_seed,
    )
    return metadata, predictions, series_fold[metadata["series_position"]]


def cross_validate(args: argparse.Namespace) -> dict[str, object]:
    metadata, predictions, row_fold = _load_inputs(args)
    sampled = deterministic_sample_mask(
        metadata["row_id"],
        metadata["time_online"],
        args.sample_fraction,
        args.sample_seed,
    )
    directory = Path(args.model_directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = np.memmap(
        directory / "oof_prediction.f32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=(metadata["target"].size,),
    )
    output[:] = np.nan
    parameters = dict(STACK_PARAMETERS)
    parameters["num_threads"] = args.threads
    fold_results: dict[str, object] = {}

    for fold in range(args.folds):
        training_full = row_fold != fold
        training_rows = np.flatnonzero(training_full & sampled)
        held_rows = np.flatnonzero(row_fold == fold)
        x_train = stack_features(
            predictions[0][training_rows],
            predictions[1][training_rows],
            metadata["time_online"][training_rows],
        )
        x_valid = stack_features(
            predictions[0][held_rows],
            predictions[1][held_rows],
            metadata["time_online"][held_rows],
        )
        y_train = metadata["target"][training_rows]
        weights = pair_weights_from_reference(
            y_train,
            metadata["time_online"][training_rows],
            metadata["target"][training_full],
            metadata["time_online"][training_full],
        )
        fold_parameters = dict(parameters)
        fold_parameters.update(
            {
                "seed": args.model_seed + fold,
                "bagging_seed": args.model_seed + 100 + fold,
                "data_random_seed": args.model_seed + 200 + fold,
            }
        )
        training = lgb.Dataset(
            x_train,
            label=y_train,
            weight=weights,
            feature_name=list(STACK_FEATURE_NAMES),
            free_raw_data=True,
        )
        validation = lgb.Dataset(
            x_valid,
            label=metadata["target"][held_rows],
            feature_name=list(STACK_FEATURE_NAMES),
            reference=training,
            free_raw_data=True,
        )
        booster = lgb.train(
            fold_parameters,
            training,
            num_boost_round=args.num_boost_round,
            valid_sets=[validation],
            valid_names=["held_fold"],
            feval=make_tsauc_evaluator(metadata["time_online"][held_rows]),
            callbacks=[
                lgb.early_stopping(
                    args.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(args.log_every),
            ],
        )
        prediction = booster.predict(
            x_valid, num_iteration=booster.best_iteration
        ).astype(np.float32)
        output[held_rows] = prediction
        score = tsauc(
            metadata["target"][held_rows],
            prediction,
            metadata["time_online"][held_rows],
        )
        booster.save_model(
            directory / f"fold_{fold}.txt",
            num_iteration=booster.best_iteration,
        )
        fold_results[str(fold)] = {
            "score": score,
            "best_iteration": int(booster.best_iteration),
            "training_rows": int(training_rows.size),
            "held_rows": int(held_rows.size),
        }
        print(
            f"stack fold {fold}: TS-AUC={score:.6f}, "
            f"iteration={booster.best_iteration}",
            flush=True,
        )
        del x_train, x_valid, training, validation, booster, prediction, weights
        gc.collect()

    output.flush()
    scores = [fold_results[str(fold)]["score"] for fold in range(args.folds)]
    se = fold_standard_error(scores)
    mean = float(np.mean(scores))
    result = {
        "folds": fold_results,
        "fold_scores": scores,
        "fold_mean_tsauc": mean,
        "fold_standard_deviation": float(np.std(scores, ddof=1)),
        "fold_standard_error": se,
        "fold_mean_95_percent_ci": [
            mean - 2.7764451051977987 * se,
            mean + 2.7764451051977987 * se,
        ],
        "pooled_oof_tsauc": tsauc(
            metadata["target"], output, metadata["time_online"]
        ),
    }
    (directory / "cv_results.json").write_text(json.dumps(result, indent=2))
    manifest = {
        "kind": "oof_stacker",
        "core_directory": str(Path(args.core_directory)),
        "candidate_directory": str(Path(args.candidate_directory)),
        "feature_names": list(STACK_FEATURE_NAMES),
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "sample_fraction": args.sample_fraction,
        "sample_seed": args.sample_seed,
        "parameters": parameters,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def fit_final(args: argparse.Namespace) -> Path:
    metadata, predictions, _ = _load_inputs(args)
    directory = Path(args.model_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    sampled = deterministic_sample_mask(
        metadata["row_id"],
        metadata["time_online"],
        float(manifest["sample_fraction"]),
        int(manifest["sample_seed"]),
    )
    rows = np.flatnonzero(sampled)
    features = stack_features(
        predictions[0][rows],
        predictions[1][rows],
        metadata["time_online"][rows],
    )
    target = metadata["target"][rows]
    weights = pair_weights_from_reference(
        target,
        metadata["time_online"][rows],
        metadata["target"],
        metadata["time_online"],
    )
    training = lgb.Dataset(
        features,
        label=target,
        weight=weights,
        feature_name=list(STACK_FEATURE_NAMES),
        free_raw_data=True,
    )
    results = json.loads((directory / "cv_results.json").read_text())
    iterations = int(
        round(
            np.median(
                [value["best_iteration"] for value in results["folds"].values()]
            )
        )
    )
    parameters = dict(manifest["parameters"])
    parameters["num_threads"] = args.threads
    parameters.update(
        {
            "seed": args.model_seed,
            "bagging_seed": args.model_seed + 100,
            "data_random_seed": args.model_seed + 200,
        }
    )
    booster = lgb.train(parameters, training, num_boost_round=iterations)
    path = directory / "final.txt"
    booster.save_model(path)
    manifest["final_model"] = {
        "path": path.name,
        "iterations": iterations,
        "training_rows": int(rows.size),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved final stacker: {path} ({iterations} rounds)", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("cross-validate", "fit-final"):
        child = subparsers.add_parser(command)
        child.add_argument("--cache-directory", default="models/artifacts")
        child.add_argument("--cache-name", default="core_features")
        child.add_argument(
            "--core-directory", default="models/artifacts/model_core"
        )
        child.add_argument(
            "--candidate-directory", default="models/artifacts/model_candidate"
        )
        child.add_argument(
            "--model-directory", default="models/artifacts/model_stack"
        )
        child.add_argument("--folds", type=int, default=5)
        child.add_argument("--fold-seed", type=int, default=73051)
        child.add_argument("--sample-fraction", type=float, default=0.20)
        child.add_argument("--sample-seed", type=int, default=99173)
        child.add_argument("--model-seed", type=int, default=62501)
        child.add_argument(
            "--threads", type=int, default=max(1, min(8, os.cpu_count() or 1))
        )
    cv = subparsers.choices["cross-validate"]
    cv.add_argument("--num-boost-round", type=int, default=500)
    cv.add_argument("--early-stopping-rounds", type=int, default=60)
    cv.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "cross-validate":
        cross_validate(args)
    else:
        fit_final(args)


if __name__ == "__main__":
    main()
