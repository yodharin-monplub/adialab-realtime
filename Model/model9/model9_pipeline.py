"""Development-only full-row scaling and exploratory stacking for Model 9.

Model 9 never indexes the already revealed Model 6 outer holdout. Every base
prediction is produced by a voter that excludes the scored series. Architecture
families nevertheless inherit Model 5 banks selected on these same development
folds, so fixed-architecture raw CV is reused-development, conditional evidence.
Postprocessing, promotion, blending, and stacking reuse cross-fitted predictions
without fully nesting every base refit and are exploratory diagnostics.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from models.model4 import model4_pipeline as model4
from models.model6 import model6_pipeline as model6
from models.model6.config import make_model6_config
from models.model9.config import MODEL9_VERSION, Model9Config
from train_cv import matrix_rows, pair_weights_from_reference
from tsauc import fold_standard_error, tsauc


@dataclass
class Model9Context:
    """Model 9 protocol plus read-only Model 6 matrices and split contracts."""

    config: Model9Config
    source: model6.Model6Context
    source_contract_hash: str


def _ready_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _ready_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        _ready_json(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_ready_json(value), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_or_validate(path: Path, value: Any) -> None:
    ready = _ready_json(value)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != ready:
            raise ValueError(
                f"{path} belongs to another Model 9 protocol; retain it and "
                "use the configuration-derived run directory"
            )
        return
    _atomic_json(path, ready)


def _mean_se(scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("fold scores must be finite and non-empty")
    return {
        "fold_scores": values.tolist(),
        "fold_mean": float(values.mean()),
        "fold_standard_deviation": (
            float(values.std(ddof=1)) if values.size > 1 else None
        ),
        "fold_standard_error": (
            float(fold_standard_error(values)) if values.size > 1 else None
        ),
    }


def robust_score(scores: Sequence[float], penalty: float) -> float:
    """Mean fold TSAUC minus the configured multiple of its standard error."""

    values = np.asarray(scores, dtype=np.float64)
    standard_error = float(fold_standard_error(values)) if values.size > 1 else 0.0
    return float(values.mean() - penalty * standard_error)


def load_context(config: Model9Config) -> Model9Context:
    """Load train-derived caches while prohibiting Model 6 holdout predictions."""

    config.validate()
    source_config = make_model6_config(config.project_root)
    if source_config.artifact_path != config.source_model6_path:
        raise ValueError("Model 9 source path does not identify the fixed Model 6 run")
    source = model6.load_context(source_config)
    frozen_path = config.source_model6_path / "frozen_contract.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Model 6 frozen development contract is missing")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    source_contract_hash = str(frozen["contract_hash"])
    baseline_path = config.source_model6_path / config.source_baseline_prediction
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Model 6 development OOF is missing: {baseline_path}")
    prohibited = config.source_model6_path / "holdout_predictions/global.f32.npy"
    manifest = {
        "experiment_version": MODEL9_VERSION,
        "protocol": config.protocol(),
        "protocol_hash": _hash_json(config.protocol()),
        "source_model6_contract_hash": source_contract_hash,
        "source_model6_selection_hash": source.source_selection["selection_hash"],
        "training_series": int(len(source.data.training)),
        "online_rows": int(source.data.training.online_row_count),
        "development_series": int(source.split.development_series_mask.sum()),
        "sealed_outer_series": int(source.split.holdout_series_mask.sum()),
        "source_baseline": str(baseline_path.relative_to(config.project_root)),
        "prohibited_holdout_prediction": str(prohibited.relative_to(config.project_root)),
        "complete_series_fold_roles": True,
        "nested_label_independent_row_sampling": True,
        "outer_holdout_read": False,
    }
    _write_or_validate(config.artifact_path / "experiment_manifest.json", manifest)
    return Model9Context(config, source, source_contract_hash)


def context_summary(context: Model9Context) -> dict[str, Any]:
    """Return a compact status without reading any outer-holdout outcome."""

    source = context.source
    return {
        "run": context.config.artifact_directory,
        "device": source.actual_device,
        "development_series": int(source.split.development_series_mask.sum()),
        "sealed_outer_series": int(source.split.holdout_series_mask.sum()),
        "development_rows": int(source.split.development_rows.size),
        "architectures": list(context.config.architectures),
        "scale_train_rows": list(context.config.scale_train_rows),
        "outer_holdout_read": False,
    }


def complete_series_cap(
    eligible_rows: np.ndarray,
    series_position: np.ndarray,
    maximum_rows: int,
    seed: int,
) -> np.ndarray:
    """Select whole series, label-independently, up to a deterministic row cap."""

    rows = np.asarray(eligible_rows, dtype=np.int64)
    owner = np.asarray(series_position, dtype=np.int64)
    if rows.ndim != 1 or owner.ndim != 1 or np.any(rows < 0) or np.any(rows >= owner.size):
        raise ValueError("eligible rows or series ownership are malformed")
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive")
    if rows.size <= maximum_rows:
        return rows.copy()
    candidates, lengths = np.unique(owner[rows], return_counts=True)
    order = np.random.default_rng(seed).permutation(candidates.size)
    cumulative = np.cumsum(lengths[order], dtype=np.int64)
    count = int(np.searchsorted(cumulative, maximum_rows, side="right"))
    if count == 0:
        count = 1
    chosen = candidates[order[:count]]
    selected_series = np.zeros(int(owner.max()) + 1, dtype=bool)
    selected_series[chosen] = True
    selected = rows[selected_series[owner[rows]]]
    # The construction filters by series membership, so this vectorized check
    # is enough to guard accidental partial-series changes without an O(S*R)
    # loop over every selected series and every eligible row.
    selected_ids, selected_lengths = np.unique(owner[selected], return_counts=True)
    expected = {int(series): int(length) for series, length in zip(candidates, lengths)}
    if set(selected_ids.tolist()) != set(chosen.tolist()) or any(
        int(length) != expected[int(series)]
        for series, length in zip(selected_ids, selected_lengths)
    ):
        raise AssertionError("complete-series sampling split a series")
    return selected


def nested_row_cap(
    eligible_rows: np.ndarray,
    maximum_rows: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic nested row subset without consulting labels.

    Complete series are assigned to fit/stop/score roles before this function
    runs. Sampling rows only inside one role preserves that series-level split
    while sampling across the eligible role. Using the same seed makes every
    smaller cap an exact subset of a larger cap.
    """

    rows = np.asarray(eligible_rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0):
        raise ValueError("eligible rows must be a non-negative one-dimensional array")
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive")
    if rows.size <= maximum_rows:
        return rows.copy()
    order = np.random.default_rng(seed).permutation(rows.size)
    return np.sort(rows[order[:maximum_rows]])


def _role_masks(context: Model9Context, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if fold not in range(context.config.folds):
        raise ValueError("fold must be in 0..4")
    row_fold = context.source.split.row_inner_fold
    stop_fold = (fold + 1) % context.config.folds
    development = row_fold >= 0
    score = development & (row_fold == fold)
    stop = development & (row_fold == stop_fold)
    fit = development & ~score & ~stop
    return fit, stop, score, stop_fold


def role_rows(
    context: Model9Context,
    fold: int,
    train_cap: int,
    stop_cap: int,
    *,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return series-disjoint fit/stop/score roles with nested row samples.

    The held score fold is always complete. Fit and stop series remain wholly
    assigned to their roles, while label-independent row subsampling makes the
    requested row scales meaningful.
    """

    fit_mask, stop_mask, score_mask, stop_fold = _role_masks(context, fold)
    seed = context.config.seed + seed_offset + fold * 10_003
    fit = nested_row_cap(np.flatnonzero(fit_mask), train_cap, seed + 1)
    stop = nested_row_cap(np.flatnonzero(stop_mask), stop_cap, seed + 2)
    score = np.flatnonzero(score_mask)
    return fit, stop, score, stop_fold


def architecture_columns(
    context: Model9Context, architecture: str, fold: int
) -> tuple[np.ndarray, list[str]]:
    """Resolve a fold-excluded feature contract into global matrix columns."""

    source = context.source
    if architecture == "candidate_all_gbdt":
        columns = source.family_columns["candidate"].copy()
        names = list(source.family_names["candidate"])
    elif architecture == "candidate_split512_gbdt":
        columns, names = model6._columns_for_contract(
            source,
            "candidate",
            fold,
            {"method": "split", "count": 512, "key": "split_512"},
        )
    elif architecture == "core_fisher512_gbdt":
        columns, names = model6._columns_for_contract(
            source,
            "core",
            fold,
            {"method": "fisher", "count": 512, "key": "fisher_512"},
        )
    elif architecture == "joint_fisher768_gbdt":
        core_columns, core_names = model6._columns_for_contract(
            source,
            "core",
            fold,
            {"method": "fisher", "count": 256, "key": "fisher_256"},
        )
        candidate_columns, candidate_names = model6._columns_for_contract(
            source,
            "candidate",
            fold,
            {"method": "fisher", "count": 512, "key": "fisher_512"},
        )
        columns = np.concatenate((core_columns, candidate_columns)).astype(np.int32)
        names = core_names + candidate_names
    else:
        raise ValueError(f"unknown Model 9 architecture: {architecture}")
    if len(names) != len(set(names)) or len(columns) != len(np.unique(columns)):
        raise ValueError(f"{architecture} emitted duplicate features")
    return np.asarray(columns, dtype=np.int32), names


def _gbdt_hyperparameters() -> dict[str, Any]:
    return {
        "name": "model5_strong_baseline",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.60,
        "bagging_fraction": 0.85,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "max_bin": 127,
    }


def _fit_base_fold(
    context: Model9Context,
    architecture: str,
    fold: int,
    train_cap: int,
    stop_cap: int,
    stage: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one score-fold-excluded LightGBM voter and checkpoint it."""

    directory = (
        context.config.artifact_path
        / stage
        / architecture
        / f"rows_{train_cap}"
    )
    report_path = directory / f"fold_{fold}.json"
    prediction_path = directory / f"fold_{fold}.prediction.f32.npy"
    model_path = directory / f"fold_{fold}.txt"
    columns_path = directory / f"fold_{fold}.columns.npz"
    columns, names = architecture_columns(context, architecture, fold)
    fit_rows, stop_rows, score_rows, stop_fold = role_rows(
        context,
        fold,
        train_cap,
        stop_cap,
        seed_offset=0 if stage == "scale_cv" else 1_000_000,
    )
    contract = {
        "experiment_version": MODEL9_VERSION,
        "stage": stage,
        "architecture": architecture,
        "score_fold": fold,
        "stop_fold": stop_fold,
        "train_cap": train_cap,
        "stop_cap": stop_cap,
        "feature_count": int(columns.size),
        "feature_names_hash": _hash_json(names),
        "column_hash": hashlib.sha256(columns.tobytes()).hexdigest(),
        "hyperparameters": _gbdt_hyperparameters(),
        "rounds": context.config.scale_rounds,
        "patience": context.config.scale_patience,
        "complete_series_fold_roles": True,
        "nested_label_independent_row_sampling": True,
        "outer_holdout_read": False,
    }
    contract_hash = _hash_json(contract)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise ValueError(f"stale Model 9 checkpoint: {report_path}")
        for required in (prediction_path, model_path, columns_path):
            if not required.is_file():
                raise ValueError(f"incomplete Model 9 checkpoint: {required}")
        prediction = np.load(prediction_path, allow_pickle=False)
        if prediction.shape != score_rows.shape:
            raise ValueError(f"checkpoint score-row count changed: {prediction_path}")
        print(
            f"{stage}/{architecture}/rows{train_cap}/fold{fold}: "
            f"checkpoint TSAUC={report['tsroc']:.6f}",
            flush=True,
        )
        return prediction, report

    print(
        f"{stage}/{architecture}/rows{train_cap}/fold{fold}: "
        f"{fit_rows.size:,} fit, {stop_rows.size:,} stop, "
        f"{score_rows.size:,} score, {columns.size:,} features",
        flush=True,
    )
    started = time.perf_counter()
    source = context.source
    metadata = source.data.metadata
    fit_mask, stop_mask, _, _ = _role_masks(context, fold)
    x_fit = model6._sanitize(matrix_rows(source.data.joint_matrix, fit_rows, columns))
    x_stop = model6._sanitize(matrix_rows(source.data.joint_matrix, stop_rows, columns))
    y_fit = metadata["target"][fit_rows]
    y_stop = metadata["target"][stop_rows]
    fit_weight = pair_weights_from_reference(
        y_fit,
        metadata["time_online"][fit_rows],
        metadata["target"][fit_mask],
        metadata["time_online"][fit_mask],
    )
    stop_weight = pair_weights_from_reference(
        y_stop,
        metadata["time_online"][stop_rows],
        metadata["target"][stop_mask],
        metadata["time_online"][stop_mask],
    )
    fitted = model6._fit_model(
        model_type="lightgbm_gbdt",
        hyperparameters=_gbdt_hyperparameters(),
        x_train=x_fit,
        y_train=y_fit,
        train_weight=fit_weight,
        x_stop=x_stop,
        y_stop=y_stop,
        stop_weight=stop_weight,
        stop_time=metadata["time_online"][stop_rows],
        rounds=context.config.scale_rounds,
        patience=context.config.scale_patience,
        seed=context.config.seed + fold * 10_003 + train_cap,
        threads=context.config.num_threads,
        device=source.actual_device,
    )
    del x_fit, y_fit, fit_weight, stop_weight
    gc.collect()
    prediction = model6._predict_feature_rows(source, fitted, score_rows, columns)
    target = metadata["target"][score_rows]
    score = float(tsauc(target, prediction, metadata["time_online"][score_rows]))
    stop_prediction = model6._predict_array(fitted, x_stop)
    stop_score = float(tsauc(y_stop, stop_prediction, metadata["time_online"][stop_rows]))
    directory.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction.astype(np.float32))
    np.savez_compressed(
        columns_path,
        columns=columns,
        names=np.asarray(names, dtype=np.str_),
    )
    fitted.model.save_model(str(model_path), num_iteration=fitted.best_iteration)
    report = {
        **contract,
        "contract_hash": contract_hash,
        "fit_rows": int(fit_rows.size),
        "fit_series": int(np.unique(metadata["series_position"][fit_rows]).size),
        "stop_rows": int(stop_rows.size),
        "stop_series": int(np.unique(metadata["series_position"][stop_rows]).size),
        "score_rows": int(score_rows.size),
        "score_series": int(np.unique(metadata["series_position"][score_rows]).size),
        "best_iteration": int(fitted.best_iteration),
        "stop_tsroc": stop_score,
        "tsroc": score,
        "seconds": float(time.perf_counter() - started),
    }
    _atomic_json(report_path, report)
    print(
        f"{stage}/{architecture}/rows{train_cap}/fold{fold}: "
        f"TSAUC={score:.6f}, iteration={fitted.best_iteration}, "
        f"{report['seconds'] / 60:.2f} min",
        flush=True,
    )
    del x_stop, y_stop, fitted, target, stop_prediction
    gc.collect()
    return prediction, report


def _fold_scores(context: Model9Context, prediction: np.ndarray) -> list[float]:
    development = context.source.split.development_rows
    fold = context.source.split.row_inner_fold[development]
    target = context.source.data.metadata["target"][development]
    online_time = context.source.data.metadata["time_online"][development]
    return [
        float(tsauc(target[fold == held], prediction[fold == held], online_time[fold == held]))
        for held in range(context.config.folds)
    ]


def _prediction_report(context: Model9Context, prediction: np.ndarray) -> dict[str, Any]:
    development = context.source.split.development_rows
    target = context.source.data.metadata["target"][development]
    online_time = context.source.data.metadata["time_online"][development]
    scores = _fold_scores(context, prediction)
    return {
        **_mean_se(scores),
        "robust_score": robust_score(scores, context.config.uncertainty_penalty),
        "pooled_tsroc": float(tsauc(target, prediction, online_time)),
    }


def _nested_select(
    context: Model9Context,
    predictions: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exclude each held score from a fold-level exploratory selector.

    This excludes the held score scalar, but it is not a fully nested base-refit
    estimate: other-fold OOF voters can have trained or stopped on the held fold.
    """

    if not predictions:
        raise ValueError("nested selection requires predictions")
    fold_scores = {name: _fold_scores(context, value) for name, value in predictions.items()}
    development = context.source.split.development_rows
    fold = context.source.split.row_inner_fold[development]
    selected = np.empty(development.size, dtype=np.float32)
    choices = []
    for held in range(context.config.folds):
        candidates = []
        for name, scores in fold_scores.items():
            training_scores = [score for index, score in enumerate(scores) if index != held]
            candidates.append(
                (
                    robust_score(training_scores, context.config.uncertainty_penalty),
                    name,
                )
            )
        _, winner = max(candidates)
        selected[fold == held] = predictions[winner][fold == held]
        choices.append({"held_fold": held, "selected": winner})
    deployment = max(
        (
            robust_score(scores, context.config.uncertainty_penalty),
            name,
        )
        for name, scores in fold_scores.items()
    )[1]
    return selected, {
        "held_choices": choices,
        "deployment_choice": deployment,
        "component_fold_scores": fold_scores,
        "nested_score": _prediction_report(context, selected),
    }


def _blend(baseline: np.ndarray, candidate: np.ndarray, weight: float) -> np.ndarray:
    margin = (1.0 - weight) * model6._safe_logit(baseline) + weight * model6._safe_logit(candidate)
    return model6._logistic(margin).astype(np.float32)


def _blend_tuning_mask(context: Model9Context) -> np.ndarray:
    source = context.source
    series_fold = source.split.inner_series_fold
    selected_series = np.zeros(series_fold.size, dtype=bool)
    rng = np.random.default_rng(context.config.seed + 700_000)
    for fold in range(context.config.folds):
        positions = np.flatnonzero(series_fold == fold)
        count = min(context.config.blend_tuning_series_per_fold, positions.size)
        selected_series[rng.choice(positions, size=count, replace=False)] = True
    development = source.split.development_rows
    return selected_series[source.data.metadata["series_position"][development]]


def _nested_blend(
    context: Model9Context,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune fold-excluded blend weights as an exploratory level-two diagnostic."""
    development = context.source.split.development_rows
    fold = context.source.split.row_inner_fold[development]
    target = context.source.data.metadata["target"][development]
    online_time = context.source.data.metadata["time_online"][development]
    tune = _blend_tuning_mask(context)
    output = np.empty(development.size, dtype=np.float32)
    held = []
    for held_fold in range(context.config.folds):
        tuning = tune & (fold != held_fold)
        scored = []
        for weight in context.config.blend_weights:
            prediction = _blend(baseline, candidate, weight)
            score = float(tsauc(target[tuning], prediction[tuning], online_time[tuning]))
            scored.append((score, -float(weight), float(weight)))
        _, _, weight = max(scored)
        prediction = _blend(baseline, candidate, weight)
        output[fold == held_fold] = prediction[fold == held_fold]
        held.append({"held_fold": held_fold, "selected_candidate_weight": weight})
    deployment_scores = []
    for weight in context.config.blend_weights:
        prediction = _blend(baseline, candidate, weight)
        score = float(tsauc(target[tune], prediction[tune], online_time[tune]))
        deployment_scores.append((score, -float(weight), float(weight)))
    deployment_weight = max(deployment_scores)[2]
    return output, {
        "held_weights": held,
        "deployment_candidate_weight": deployment_weight,
        "nested_score": _prediction_report(context, output),
    }


def _baseline(context: Model9Context) -> np.ndarray:
    path = context.config.source_model6_path / context.config.source_baseline_prediction
    prediction = np.load(path, allow_pickle=False).astype(np.float32)
    if prediction.shape != context.source.split.development_rows.shape:
        raise ValueError("Model 6 baseline no longer aligns with development rows")
    return prediction


def run_scale_cv(context: Model9Context) -> dict[str, Any]:
    """Compare nested 400k and 1.2M row samples in series-disjoint roles."""

    summary_path = context.config.artifact_path / "scale_cv/summary.json"
    development = context.source.split.development_rows
    local_fold = context.source.split.row_inner_fold[development]
    predictions: dict[str, np.ndarray] = {}
    candidates = []
    for architecture in context.config.architectures:
        for train_cap in context.config.scale_train_rows:
            candidate_id = f"{architecture}__rows_{train_cap}"
            raw = np.full(development.size, np.nan, dtype=np.float32)
            fold_records = []
            for fold in range(context.config.folds):
                prediction, record = _fit_base_fold(
                    context,
                    architecture,
                    fold,
                    train_cap,
                    context.config.scale_stop_rows,
                    "scale_cv",
                )
                raw[local_fold == fold] = prediction
                fold_records.append(record)
            if not np.all(np.isfinite(raw)):
                raise AssertionError(f"incomplete Model 9 OOF: {candidate_id}")
            directory = context.config.artifact_path / "scale_cv" / architecture / f"rows_{train_cap}"
            np.save(directory / "raw_oof.f32.npy", raw)
            nested, postprocess = model6._postprocess_oof(context.source, raw)
            np.save(directory / "nested_oof.f32.npy", nested)
            report = {
                "candidate": candidate_id,
                "architecture": architecture,
                "train_cap": train_cap,
                "raw": _prediction_report(context, raw),
                "raw_score_status": (
                    "reused-development conditional fixed-architecture CV; "
                    "Model 5 bank selection reused these folds"
                ),
                "postprocessed": _prediction_report(context, nested),
                "postprocessed_score_status": (
                    "exploratory; transform selection is not a fully nested base refit"
                ),
                "postprocess_contract": postprocess,
                "fold_records": fold_records,
                "prediction_path": str(
                    (directory / "nested_oof.f32.npy").relative_to(context.config.project_root)
                ),
            }
            _atomic_json(directory / "summary.json", report)
            candidates.append(report)
            predictions[candidate_id] = nested
            del raw
            gc.collect()

    selected, selection = _nested_select(context, predictions)
    baseline = _baseline(context)
    blended, blend = _nested_blend(context, baseline, selected)
    output_directory = context.config.artifact_path / "scale_cv"
    np.save(output_directory / "nested_selected.f32.npy", selected)
    np.save(output_directory / "nested_baseline_blend.f32.npy", blended)
    report = {
        "stage": "model9_scale_cv",
        "selection_metric": "mean fold TSAUC - 0.5 * SE",
        "raw_fixed_architecture_status": "reused-development conditional evidence",
        "adaptive_stage_status": "exploratory; not fully nested",
        "upstream_model5_bank_selection_reused_development_folds": True,
        "not_fully_nested_base_refit": True,
        "outer_holdout_read": False,
        "baseline": _prediction_report(context, baseline),
        "candidates": sorted(
            candidates,
            key=lambda row: -row["postprocessed"]["robust_score"],
        ),
        "nested_architecture_selection": selection,
        "nested_baseline_blend": blend,
    }
    _atomic_json(summary_path, report)
    return report


def _promotion_contract(
    context: Model9Context, scale: dict[str, Any]
) -> dict[str, Any]:
    """Freeze exploratory scale-stage winners before promoted scoring.

    Each held fold receives the architecture selected from the other four
    scale-stage folds.  The union is fitted for checkpoint reuse, but a
    promoted-fold score is never allowed to choose its own architecture.
    """

    largest = context.config.scale_train_rows[-1]
    rows = [row for row in scale["candidates"] if row["train_cap"] == largest]
    score_by_architecture = {
        row["architecture"]: row["postprocessed"]["fold_scores"] for row in rows
    }
    selected = set()
    held_choices = []
    for held in range(context.config.folds):
        winner = max(
            (
                robust_score(
                    [score for index, score in enumerate(scores) if index != held],
                    context.config.uncertainty_penalty,
                ),
                architecture,
            )
            for architecture, scores in score_by_architecture.items()
        )[1]
        selected.add(winner)
        held_choices.append({"held_fold": held, "selected": winner})
    deployment = max(
        (
            robust_score(scores, context.config.uncertainty_penalty),
            architecture,
        )
        for architecture, scores in score_by_architecture.items()
    )[1]
    selected.add(deployment)
    if len(selected) > context.config.maximum_promoted_architectures:
        raise ValueError("nested promotion selected more architectures than allowed")
    return {
        "source_train_rows": largest,
        "selection_metric": "mean fold TSAUC - 0.5 * SE",
        "held_choices": held_choices,
        "deployment_choice": deployment,
        "fit_architectures": list(sorted(selected)),
    }


def run_promoted_cv(context: Model9Context) -> dict[str, Any]:
    """Retrain exploratory fold-excluded promotions on near-full row samples."""

    scale = run_scale_cv(context)
    promotion = _promotion_contract(context, scale)
    promoted = tuple(promotion["fit_architectures"])
    development = context.source.split.development_rows
    local_fold = context.source.split.row_inner_fold[development]
    predictions: dict[str, np.ndarray] = {}
    candidates = []
    for architecture in promoted:
        raw = np.full(development.size, np.nan, dtype=np.float32)
        fold_records = []
        for fold in range(context.config.folds):
            prediction, record = _fit_base_fold(
                context,
                architecture,
                fold,
                context.config.promoted_train_rows,
                context.config.promoted_stop_rows,
                "promoted_cv",
            )
            raw[local_fold == fold] = prediction
            fold_records.append(record)
        directory = context.config.artifact_path / "promoted_cv" / architecture / f"rows_{context.config.promoted_train_rows}"
        np.save(directory / "raw_oof.f32.npy", raw)
        nested, postprocess = model6._postprocess_oof(context.source, raw)
        np.save(directory / "nested_oof.f32.npy", nested)
        report = {
            "architecture": architecture,
            "train_cap": context.config.promoted_train_rows,
            "raw": _prediction_report(context, raw),
            "raw_score_status": (
                "reused-development conditional fixed-architecture CV; "
                "Model 5 bank selection reused these folds"
            ),
            "postprocessed": _prediction_report(context, nested),
            "postprocessed_score_status": (
                "exploratory; transform selection is not a fully nested base refit"
            ),
            "postprocess_contract": postprocess,
            "fold_records": fold_records,
            "prediction_path": str(
                (directory / "nested_oof.f32.npy").relative_to(context.config.project_root)
            ),
        }
        _atomic_json(directory / "summary.json", report)
        predictions[architecture] = nested
        candidates.append(report)
        del raw
        gc.collect()

    # Apply the choices frozen from scale CV. Do not reselect using promoted
    # predictions: doing so would let a promoted score fold choose itself.
    selected = np.empty(development.size, dtype=np.float32)
    for choice in promotion["held_choices"]:
        held = int(choice["held_fold"])
        architecture = str(choice["selected"])
        selected[local_fold == held] = predictions[architecture][local_fold == held]
    selection = {
        "source": "scale_cv choices selected without each held fold",
        "held_choices": promotion["held_choices"],
        "deployment_choice": promotion["deployment_choice"],
        "nested_score": _prediction_report(context, selected),
    }
    baseline = _baseline(context)
    blended, blend = _nested_blend(context, baseline, selected)
    directory = context.config.artifact_path / "promoted_cv"
    np.save(directory / "nested_selected.f32.npy", selected)
    np.save(directory / "nested_baseline_blend.f32.npy", blended)
    report = {
        "stage": "model9_promoted_cv",
        "promotion_source": "largest scale selected without each held fold",
        "evidence_status": "exploratory; not fully nested",
        "upstream_model5_bank_selection_reused_development_folds": True,
        "not_fully_nested_base_refit": True,
        "promotion_contract": promotion,
        "promoted_architectures": list(promoted),
        "outer_holdout_read": False,
        "baseline": _prediction_report(context, baseline),
        "candidates": candidates,
        "nested_architecture_selection": selection,
        "nested_baseline_blend": blend,
    }
    _atomic_json(directory / "summary.json", report)
    return report


def meta_features(
    probabilities: Mapping[str, np.ndarray], online_time: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Create finite OOF-logit, disagreement, and smooth age interactions."""

    names = sorted(probabilities)
    if len(names) < 2:
        raise ValueError("meta stack requires at least two base predictions")
    arrays = [np.asarray(probabilities[name], dtype=np.float64) for name in names]
    size = arrays[0].size
    if any(array.shape != (size,) for array in arrays):
        raise ValueError("meta prediction arrays are not aligned")
    margins = np.column_stack([model6._safe_logit(array) for array in arrays])
    age = np.clip(np.log1p(np.asarray(online_time, dtype=np.float64)) / np.log1p(1024.0), 0.0, 2.0)
    mean = margins.mean(axis=1)
    standard_deviation = margins.std(axis=1)
    spread = margins.max(axis=1) - margins.min(axis=1)
    matrix = np.column_stack(
        (
            margins,
            mean,
            standard_deviation,
            spread,
            age,
            margins * age[:, None],
            standard_deviation * age,
        )
    ).astype(np.float32)
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=25.0, neginf=-25.0)
    np.clip(matrix, -25.0, 25.0, out=matrix)
    feature_names = (
        [f"logit__{name}" for name in names]
        + ["logit_mean", "logit_std", "logit_spread", "log_age"]
        + [f"logit_x_age__{name}" for name in names]
        + ["disagreement_x_age"]
    )
    return np.ascontiguousarray(matrix), feature_names


def _meta_hyperparameters() -> dict[str, Any]:
    return {
        "name": "shallow_oof_gate",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 1500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "lambda_l1": 1.0,
        "lambda_l2": 12.0,
        "max_bin": 127,
    }


def run_meta_stack(context: Model9Context) -> dict[str, Any]:
    """Fit an exploratory gate over OOF scores without fully nested base refits."""

    promoted = run_promoted_cv(context)
    probabilities: dict[str, np.ndarray] = {"model6_global": _baseline(context)}
    for candidate in promoted["candidates"]:
        probabilities[candidate["architecture"]] = np.load(
            context.config.project_root / candidate["prediction_path"],
            allow_pickle=False,
        ).astype(np.float32)
    probabilities["model9_nested_blend"] = np.load(
        context.config.artifact_path / "promoted_cv/nested_baseline_blend.f32.npy",
        allow_pickle=False,
    ).astype(np.float32)
    source = context.source
    development = source.split.development_rows
    online_time = source.data.metadata["time_online"][development]
    target = source.data.metadata["target"][development]
    fold = source.split.row_inner_fold[development]
    matrix, feature_names = meta_features(probabilities, online_time)
    global_to_local = np.full(source.data.metadata["target"].size, -1, dtype=np.int64)
    global_to_local[development] = np.arange(development.size)
    oof = np.full(development.size, np.nan, dtype=np.float32)
    fold_records = []
    directory = context.config.artifact_path / "meta_stack"
    for held in range(context.config.folds):
        report_path = directory / f"fold_{held}.json"
        prediction_path = directory / f"fold_{held}.prediction.f32.npy"
        model_path = directory / f"fold_{held}.txt"
        fit_global, stop_global, score_global, stop_fold = role_rows(
            context,
            held,
            context.config.meta_train_rows,
            context.config.meta_stop_rows,
            seed_offset=2_000_000,
        )
        fit_rows = global_to_local[fit_global]
        stop_rows = global_to_local[stop_global]
        score_rows = global_to_local[score_global]
        if np.any(np.concatenate((fit_rows, stop_rows, score_rows)) < 0):
            raise AssertionError("meta role escaped development rows")
        contract = {
            "experiment_version": MODEL9_VERSION,
            "stage": "meta_stack",
            "score_fold": held,
            "stop_fold": stop_fold,
            "feature_names": feature_names,
            "base_components": sorted(probabilities),
            "fit_cap": context.config.meta_train_rows,
            "stop_cap": context.config.meta_stop_rows,
            "hyperparameters": _meta_hyperparameters(),
            "outer_holdout_read": False,
        }
        contract_hash = _hash_json(contract)
        if report_path.exists():
            record = json.loads(report_path.read_text(encoding="utf-8"))
            if record.get("contract_hash") != contract_hash:
                raise ValueError(f"stale Model 9 meta checkpoint: {report_path}")
            prediction = np.load(prediction_path, allow_pickle=False)
        else:
            print(
                f"meta/fold{held}: {fit_rows.size:,} fit, {stop_rows.size:,} "
                f"stop, {score_rows.size:,} score, {matrix.shape[1]} features",
                flush=True,
            )
            started = time.perf_counter()
            fit_mask = (fold != held) & (fold != stop_fold)
            stop_mask = fold == stop_fold
            fit_weight = pair_weights_from_reference(
                target[fit_rows], online_time[fit_rows], target[fit_mask], online_time[fit_mask]
            )
            stop_weight = pair_weights_from_reference(
                target[stop_rows], online_time[stop_rows], target[stop_mask], online_time[stop_mask]
            )
            fitted = model6._fit_model(
                model_type="lightgbm_gbdt",
                hyperparameters=_meta_hyperparameters(),
                x_train=matrix[fit_rows],
                y_train=target[fit_rows],
                train_weight=fit_weight,
                x_stop=matrix[stop_rows],
                y_stop=target[stop_rows],
                stop_weight=stop_weight,
                stop_time=online_time[stop_rows],
                rounds=context.config.meta_rounds,
                patience=context.config.meta_patience,
                seed=context.config.seed + 3_000_000 + held,
                threads=context.config.num_threads,
                device=source.actual_device,
            )
            prediction = model6._predict_array(fitted, matrix[score_rows]).astype(np.float32)
            score = float(tsauc(target[score_rows], prediction, online_time[score_rows]))
            directory.mkdir(parents=True, exist_ok=True)
            np.save(prediction_path, prediction)
            fitted.model.save_model(str(model_path), num_iteration=fitted.best_iteration)
            record = {
                **contract,
                "contract_hash": contract_hash,
                "fit_rows": int(fit_rows.size),
                "stop_rows": int(stop_rows.size),
                "score_rows": int(score_rows.size),
                "best_iteration": int(fitted.best_iteration),
                "tsroc": score,
                "seconds": float(time.perf_counter() - started),
            }
            _atomic_json(report_path, record)
            print(
                f"meta/fold{held}: TSAUC={score:.6f}, {record['seconds'] / 60:.2f} min",
                flush=True,
            )
            del fitted, fit_weight, stop_weight
            gc.collect()
        oof[fold == held] = prediction
        fold_records.append(record)
    if not np.all(np.isfinite(oof)):
        raise AssertionError("Model 9 meta OOF is incomplete")
    np.save(directory / "raw_oof.f32.npy", oof)
    nested, postprocess = model6._postprocess_oof(source, oof)
    np.save(directory / "nested_oof.f32.npy", nested)

    final_components = {
        "model6_global": probabilities["model6_global"],
        "model9_promoted_blend": probabilities["model9_nested_blend"],
        "model9_meta": nested,
    }
    primary, selection = _nested_select(context, final_components)
    np.save(directory / "nested_primary.f32.npy", primary)
    owner = source.data.metadata["series_position"][development]
    bootstrap = model4.cluster_bootstrap_tsauc(
        target,
        primary,
        online_time,
        owner,
        replicates=context.config.bootstrap_replicates,
        seed=context.config.bootstrap_seed,
    )
    report = {
        "stage": "model9_meta_stack",
        "evidence_status": "exploratory level-two OOF; not fully nested",
        "upstream_model5_bank_selection_reused_development_folds": True,
        "not_fully_nested_base_refit": True,
        "outer_holdout_read": False,
        "feature_names": feature_names,
        "base_components": sorted(probabilities),
        "baseline": _prediction_report(context, probabilities["model6_global"]),
        "promoted_blend": _prediction_report(context, probabilities["model9_nested_blend"]),
        "meta_raw": _prediction_report(context, oof),
        "meta_postprocessed": _prediction_report(context, nested),
        "postprocess_contract": postprocess,
        "primary_nested_selection": selection,
        "primary": _prediction_report(context, primary),
        "primary_complete_series_bootstrap": bootstrap,
        "fold_records": fold_records,
        "target_tsroc": context.config.target_tsroc,
        "target_reached": bool(
            _prediction_report(context, primary)["fold_mean"] >= context.config.target_tsroc
        ),
        "target_reached_is_exploratory": True,
    }
    _atomic_json(directory / "summary.json", report)
    del matrix, global_to_local, oof, nested, primary
    gc.collect()
    return report


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return CLI-friendly metrics without multi-megabyte grids."""

    if report.get("stage") == "model9_scale_cv":
        return {
            "stage": report["stage"],
            "raw_fixed_architecture_status": report.get(
                "raw_fixed_architecture_status",
                "reused-development conditional evidence",
            ),
            "adaptive_stage_status": report.get(
                "adaptive_stage_status", "exploratory; not fully nested"
            ),
            "baseline": report["baseline"],
            "candidates": [
                {
                    "candidate": row["candidate"],
                    "raw": row["raw"],
                    "raw_score_status": row.get(
                        "raw_score_status", "reused-development conditional evidence"
                    ),
                    "postprocessed": row["postprocessed"],
                    "postprocessed_score_status": row.get(
                        "postprocessed_score_status",
                        "exploratory; not fully nested",
                    ),
                }
                for row in report["candidates"]
            ],
            "nested_selection": report["nested_architecture_selection"]["nested_score"],
            "nested_baseline_blend": report["nested_baseline_blend"]["nested_score"],
            "outer_holdout_read": False,
        }
    if report.get("stage") == "model9_promoted_cv":
        return {
            "stage": report["stage"],
            "evidence_status": report.get(
                "evidence_status", "exploratory; not fully nested"
            ),
            "promoted_architectures": report["promoted_architectures"],
            "candidates": [
                {
                    "architecture": row["architecture"],
                    "raw": row["raw"],
                    "raw_score_status": row.get(
                        "raw_score_status", "reused-development conditional evidence"
                    ),
                    "postprocessed": row["postprocessed"],
                    "postprocessed_score_status": row.get(
                        "postprocessed_score_status",
                        "exploratory; not fully nested",
                    ),
                }
                for row in report["candidates"]
            ],
            "nested_selection": report["nested_architecture_selection"]["nested_score"],
            "nested_baseline_blend": report["nested_baseline_blend"]["nested_score"],
            "outer_holdout_read": False,
        }
    return {
        "stage": report.get("stage"),
        "evidence_status": report.get(
            "evidence_status", "exploratory; not fully nested"
        ),
        "baseline": report.get("baseline"),
        "promoted_blend": report.get("promoted_blend"),
        "meta_postprocessed": report.get("meta_postprocessed"),
        "primary": report.get("primary"),
        "bootstrap": report.get("primary_complete_series_bootstrap"),
        "target_reached": report.get("target_reached"),
        "target_reached_is_exploratory": report.get(
            "target_reached_is_exploratory", True
        ),
        "outer_holdout_read": False,
    }


def run_all(context: Model9Context) -> dict[str, Any]:
    """Run or resume every development-only Model 9 stage."""

    run_scale_cv(context)
    run_promoted_cv(context)
    return run_meta_stack(context)


__all__ = [
    "Model9Context",
    "architecture_columns",
    "compact_report",
    "complete_series_cap",
    "context_summary",
    "load_context",
    "meta_features",
    "nested_row_cap",
    "role_rows",
    "robust_score",
    "run_all",
    "run_meta_stack",
    "run_promoted_cv",
    "run_scale_cv",
]
