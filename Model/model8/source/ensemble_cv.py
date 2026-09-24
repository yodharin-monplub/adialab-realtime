"""Evaluate honest out-of-fold ensembles under the exact TS-AUC metric.

Each component directory must be produced by ``train_cv.py`` with the same
series folds.  Predictions are combined only where every component came from
a model that held that entire series out.  For two components, the optional
weight search is nested across folds: the weight applied to fold k is chosen
using the other completed folds, never fold k itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_pipeline import load_feature_cache
from train_cv import make_series_folds
from tsauc import fold_standard_error, tsauc


def safe_logit(probability: np.ndarray) -> np.ndarray:
    """Convert probabilities to finite margins for scale-robust averaging."""

    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def blend_predictions(
    predictions: Sequence[np.ndarray],
    weights: Sequence[float],
    method: str = "logit",
) -> np.ndarray:
    """Return a convex probability- or logit-space ensemble score."""

    if len(predictions) != len(weights) or not predictions:
        raise ValueError("predictions and weights must have the same nonzero length")
    weight = np.asarray(weights, dtype=np.float64)
    if np.any(weight < 0) or not np.isfinite(weight).all() or weight.sum() <= 0:
        raise ValueError("ensemble weights must be finite and nonnegative")
    weight /= weight.sum()
    arrays = [np.asarray(value, dtype=np.float64) for value in predictions]
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("component prediction shapes differ")
    if method == "logit":
        arrays = [safe_logit(value) for value in arrays]
    elif method != "probability":
        raise ValueError("method must be 'logit' or 'probability'")
    return sum(coefficient * value for coefficient, value in zip(weight, arrays))


def _mean_se_interval(scores: Sequence[float]) -> dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    mean = float(values.mean())
    se = fold_standard_error(values)
    if values.size == 5:
        critical = 2.7764451051977987
    elif values.size > 1:
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.975, df=values.size - 1))
    else:
        critical = math.nan
    return {
        "fold_scores": values.tolist(),
        "fold_mean_tsauc": mean,
        "fold_standard_deviation": (
            float(values.std(ddof=1)) if values.size > 1 else math.nan
        ),
        "fold_standard_error": se,
        "fold_mean_95_percent_ci": [mean - critical * se, mean + critical * se],
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Evaluate fixed and nested-selected blends and write a JSON report."""

    directories = [Path(value) for value in args.model_directories]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in directories]
    fold_counts = {int(value["folds"]) for value in manifests}
    fold_seeds = {int(value["fold_seed"]) for value in manifests}
    if len(fold_counts) != 1 or len(fold_seeds) != 1:
        raise ValueError("component models do not use identical fold definitions")
    n_folds = fold_counts.pop()
    fold_seed = fold_seeds.pop()

    cache = load_feature_cache(args.cache_directory, args.cache_name)
    metadata = cache.load_metadata()
    series_fold = make_series_folds(
        metadata["online_lengths"],
        metadata["tau_index"],
        n_splits=n_folds,
        seed=fold_seed,
    )
    row_fold = series_fold[metadata["series_position"]]
    row_count = metadata["target"].size

    component_results = []
    for path in directories:
        progress_path = path / "cv_progress.json"
        results_path = path / "cv_results.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            component_results.append(
                {"completed_folds": [int(key) for key in progress["folds"]]}
            )
        else:
            component_results.append(json.loads(results_path.read_text()))
    completed = sorted(
        set.intersection(
            *[set(value.get("completed_folds", [])) for value in component_results]
        )
    )
    if not completed:
        raise ValueError("components have no commonly completed folds")
    predictions = [
        np.memmap(
            path / "oof_prediction.f32.mmap",
            dtype=np.float32,
            mode="r",
            shape=(row_count,),
        )
        for path in directories
    ]
    eligible = np.isin(row_fold, completed)
    if any(not np.all(np.isfinite(value[eligible])) for value in predictions):
        raise ValueError("a component has missing predictions in a completed fold")

    if args.weights is None:
        fixed_weights = np.ones(len(predictions), dtype=np.float64)
    else:
        fixed_weights = np.asarray(args.weights, dtype=np.float64)
    fixed = blend_predictions(predictions, fixed_weights, args.method)
    fixed_scores = [
        tsauc(
            metadata["target"][row_fold == fold],
            fixed[row_fold == fold],
            metadata["time_online"][row_fold == fold],
        )
        for fold in completed
    ]
    result: dict[str, object] = {
        "components": [str(path) for path in directories],
        "method": args.method,
        "completed_folds": completed,
        "fixed_weights": (fixed_weights / fixed_weights.sum()).tolist(),
        "fixed": _mean_se_interval(fixed_scores),
        "fixed_pooled_oof_tsauc": tsauc(
            metadata["target"][eligible],
            fixed[eligible],
            metadata["time_online"][eligible],
        ),
    }

    if len(predictions) == 2 and len(completed) >= 3 and args.grid_step > 0:
        grid = np.arange(0.0, 1.0 + args.grid_step / 2, args.grid_step)
        fold_grid = np.empty((len(completed), grid.size), dtype=np.float64)
        for column, first_weight in enumerate(grid):
            candidate = blend_predictions(
                predictions,
                (first_weight, 1.0 - first_weight),
                args.method,
            )
            for row, fold in enumerate(completed):
                held = row_fold == fold
                fold_grid[row, column] = tsauc(
                    metadata["target"][held],
                    candidate[held],
                    metadata["time_online"][held],
                )

        selected_weights: list[float] = []
        nested_scores: list[float] = []
        for held_row in range(len(completed)):
            training_rows = np.arange(len(completed)) != held_row
            best = int(np.argmax(fold_grid[training_rows].mean(axis=0)))
            selected_weights.append(float(grid[best]))
            nested_scores.append(float(fold_grid[held_row, best]))
        global_best = int(np.argmax(fold_grid.mean(axis=0)))
        result["nested_weight_search"] = {
            "grid_step": args.grid_step,
            "first_component_weight_by_held_fold": selected_weights,
            "deployment_first_component_weight": float(np.median(selected_weights)),
            "globally_selected_first_component_weight": float(grid[global_best]),
            "globally_selected_fold_mean": float(fold_grid[:, global_best].mean()),
            **_mean_se_interval(nested_scores),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directories", nargs="+")
    parser.add_argument("--cache-directory", default="models/artifacts")
    parser.add_argument("--cache-name", default="core_features")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--method", choices=("logit", "probability"), default="logit")
    parser.add_argument("--grid-step", type=float, default=0.025)
    parser.add_argument(
        "--output", default="models/artifacts/ensemble_cv_results.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
