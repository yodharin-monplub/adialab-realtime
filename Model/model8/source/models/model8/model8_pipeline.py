"""Leakage-controlled evaluation of the six Model 8 detector banks.

The previously revealed Model 6 outer holdout is prohibited.  Every reported
row comes from Model 6's development population and is predicted by a model
that excluded the complete owning series.  Detector and blend selection for a
held fold uses only the other four development folds.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np

from feature_pipeline import (
    MODEL8_BANKS,
    FeatureCache,
    _feature_signature,
    build_feature_cache,
    feature_names,
    load_feature_cache,
)
from models.model4 import model4_pipeline as model4
from models.model6 import model6_pipeline as model6
from models.model6.config import make_model6_config
from models.model6.model6_pipeline import Model6Context, load_context as load_model6_context
from models.model8.config import MODEL8_VERSION, Model8Config
from train_cv import matrix_rows, pair_weights_from_reference, predict_rows, selected_columns
from tsauc import fold_standard_error, tsauc


@dataclass
class Model8Context:
    """Aligned Model 6 source split, new cache, and cross-fitted baseline."""

    config: Model8Config
    source: Model6Context
    development_rows: np.ndarray
    local_fold: np.ndarray
    baseline: np.ndarray
    cache: FeatureCache | None = None
    matrix: np.memmap | None = None


def _ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(item) for item in value]
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
    payload = json.dumps(_ready(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_ready(value), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_or_validate(path: Path, value: Any) -> None:
    ready = _ready(value)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != ready:
            raise ValueError(f"{path} belongs to a different Model 8 contract")
        return
    _atomic_json(path, ready)


def _mean_se(scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "fold_scores": values.tolist(),
        "fold_mean": float(values.mean()),
        "fold_standard_deviation": float(values.std(ddof=1)) if values.size > 1 else None,
        "fold_standard_error": float(fold_standard_error(values)) if values.size > 1 else None,
    }


def robust_score(scores: Sequence[float], penalty: float) -> float:
    values = np.asarray(scores, dtype=np.float64)
    se = float(fold_standard_error(values)) if values.size > 1 else 0.0
    return float(values.mean() - penalty * se)


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(bounded) - np.log1p(-bounded)


def _logistic(margin: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(margin), -35.0, 35.0)))


def _blend(
    baseline: np.ndarray, candidate: np.ndarray, candidate_weight: float
) -> np.ndarray:
    margin = (1.0 - candidate_weight) * _safe_logit(baseline)
    margin += candidate_weight * _safe_logit(candidate)
    return _logistic(margin).astype(np.float32)


def build_model8_cache(config: Model8Config) -> FeatureCache:
    """Build/resume the label-free 239-column cache from ``train/`` only."""

    config.validate()
    source = load_model6_context(make_model6_config(config.project_root))
    columns = len(feature_names(config.detector_banks))
    expected = source.data.training.online_row_count * columns * 4
    config.cache_path.mkdir(parents=True, exist_ok=True)
    manifest = config.cache_path / f"{config.cache_name}.manifest.json"
    if not manifest.exists():
        available = shutil.disk_usage(config.cache_path).free
        if expected + 3 * 1024**3 > available:
            raise OSError(
                f"Model 8 cache needs {expected / 1024**3:.1f} GiB plus a 3 GiB reserve; "
                f"only {available / 1024**3:.1f} GiB is free"
            )
    return build_feature_cache(
        source.data.training,
        cache_directory=config.cache_path,
        name=config.cache_name,
        banks=config.detector_banks,
        workers=config.cache_workers,
    )


def load_context(config: Model8Config, *, require_cache: bool = False) -> Model8Context:
    """Load aligned source artifacts without reading Model 6 holdout outcomes."""

    config.validate()
    source_config = make_model6_config(config.project_root)
    if source_config.artifact_path != config.source_model6_path:
        raise ValueError("configured source Model 6 directory is not its immutable run")
    source = load_model6_context(source_config)
    development_rows = source.split.development_rows
    local_fold = source.split.row_inner_fold[development_rows]
    if set(np.unique(local_fold)) != set(range(config.folds)):
        raise ValueError("Model 6 development folds do not match Model 8")

    baseline_path = config.source_model6_path / config.source_baseline_prediction
    if not baseline_path.exists():
        raise FileNotFoundError(f"cross-fitted baseline not found: {baseline_path}")
    baseline = np.load(baseline_path, mmap_mode="r")
    if baseline.shape != (development_rows.size,) or not np.all(np.isfinite(baseline)):
        raise ValueError("Model 6 baseline is not aligned to development rows")

    cache = None
    matrix = None
    if require_cache:
        cache = load_feature_cache(config.cache_path, config.cache_name)
        if cache.banks != config.detector_banks:
            raise ValueError("Model 8 cache bank order differs from its contract")
        cached_metadata = cache.load_metadata()
        for key, expected in source.data.metadata.items():
            if key in cached_metadata and not np.array_equal(cached_metadata[key], expected):
                raise ValueError(f"Model 8 cache is not source-aligned ({key})")
        matrix = cache.open("r")

    manifest_contract = {
        "experiment_version": MODEL8_VERSION,
        "protocol": config.protocol(),
        "protocol_hash": _hash_json(config.protocol()),
        "source_model6_manifest": str(
            (config.source_model6_path / "experiment_manifest.json").relative_to(
                config.project_root
            )
        ),
        "source_baseline": str(baseline_path.relative_to(config.project_root)),
        "source_baseline_sha256": _hash_file(baseline_path),
        "new_feature_signature": _feature_signature(config.detector_banks),
        "development_series": int(source.split.development_series_mask.sum()),
        "development_rows": int(development_rows.size),
        "prohibited_holdout_series": int(source.split.holdout_series_mask.sum()),
        "prohibited_holdout_rows": int(source.split.holdout_row_mask.sum()),
        "holdout_prediction_paths_read": False,
        "holdout_labels_used_for_fit_selection_or_score": False,
    }
    _write_or_validate(config.artifact_path / "experiment_manifest.json", manifest_contract)
    return Model8Context(
        config=config,
        source=source,
        development_rows=development_rows,
        local_fold=local_fold,
        baseline=np.asarray(baseline, dtype=np.float32),
        cache=cache,
        matrix=matrix,
    )


def context_summary(context: Model8Context) -> dict[str, Any]:
    metadata = context.source.data.metadata
    target = metadata["target"][context.development_rows]
    online_time = metadata["time_online"][context.development_rows]
    outer_result_path = context.config.artifact_path / "outer_holdout/results.json"
    outer_result = (
        json.loads(outer_result_path.read_text(encoding="utf-8"))
        if outer_result_path.exists()
        else None
    )
    return {
        "stage": "model8_status",
        "run": context.config.artifact_directory,
        "new_detector_banks": list(context.config.detector_banks),
        "new_feature_count": len(feature_names(context.config.detector_banks)),
        "development_series": int(context.source.split.development_series_mask.sum()),
        "development_rows": int(context.development_rows.size),
        "prohibited_holdout_series": int(context.source.split.holdout_series_mask.sum()),
        "cross_fitted_baseline_tsroc": float(tsauc(target, context.baseline, online_time)),
        "cache_ready": context.cache is not None,
        "development_selection_used_outer_holdout": False,
        "outer_holdout_read": outer_result is not None,
        "outer_holdout_accessed": outer_result is not None,
        "outer_holdout_tsroc": (
            outer_result["primary_outer_tsroc"] if outer_result is not None else None
        ),
    }


def role_rows(context: Model8Context, score_fold: int) -> tuple[np.ndarray, ...]:
    """Return disjoint complete-series fit, stop, and score roles."""

    stop_fold = (score_fold + 1) % context.config.folds
    fit_local = (context.local_fold != score_fold) & (context.local_fold != stop_fold)
    stop_local = context.local_fold == stop_fold
    score_local = context.local_fold == score_fold
    roles = (
        context.development_rows[fit_local],
        context.development_rows[stop_local],
        context.development_rows[score_local],
    )
    prohibited = context.source.split.holdout_row_mask
    if any(np.any(prohibited[rows]) for rows in roles):
        raise AssertionError("Model 6 holdout entered a Model 8 role")
    owner = context.source.data.metadata["series_position"]
    sets = [set(np.unique(owner[rows])) for rows in roles]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Model 8 roles share a complete series")
    return (*roles, stop_fold)


def sample_complete_series_rows(
    context: Model8Context, rows: np.ndarray, maximum_rows: int, seed: int
) -> np.ndarray:
    """Apply a deterministic, label-independent cap without splitting series."""

    rows = np.asarray(rows, dtype=np.int64)
    if rows.size <= maximum_rows:
        return rows
    metadata = context.source.data.metadata
    row_owner = metadata["series_position"][rows]
    series = np.unique(row_owner)
    shuffled = series.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    cumulative = np.cumsum(metadata["online_lengths"][shuffled], dtype=np.int64)
    count = max(1, int(np.searchsorted(cumulative, maximum_rows, side="right")))
    chosen = np.sort(shuffled[:count])
    selected = rows[np.isin(row_owner, chosen)]
    if set(np.unique(metadata["series_position"][selected])) != set(chosen):
        raise AssertionError("complete-series cap lost a selected series")
    return selected


def _fit_paths(context: Model8Context, stage: str, label: str, fold: int) -> tuple[Path, ...]:
    directory = context.config.artifact_path / stage / label
    return (
        directory / f"fold_{fold}.json",
        directory / f"fold_{fold}.prediction.f32.npy",
        directory / f"fold_{fold}.txt",
    )


def _lightgbm_parameters(
    context: Model8Context,
    *,
    fold: int,
    bank_count: int,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Return the fixed learner contract used by CV and deployment voters."""

    base_seed = context.config.seed + seed_offset + 31 * fold
    return {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 120,
        "min_sum_hessian_in_leaf": 1e-3,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "verbosity": -1,
        "num_threads": context.config.num_threads,
        "seed": base_seed + bank_count,
        "feature_fraction_seed": base_seed + 1,
        "bagging_seed": base_seed + 2,
        "data_random_seed": base_seed + 3,
        "deterministic": True,
        "force_col_wise": True,
    }


def fit_fold(
    context: Model8Context,
    *,
    stage: str,
    label: str,
    banks: Sequence[str],
    fold: int,
    train_cap: int,
    stop_cap: int,
    rounds: int,
    patience: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one LightGBM model with a score-fold-independent early-stop role."""

    if context.cache is None or context.matrix is None:
        raise ValueError("fit_fold requires the Model 8 cache")
    selected_banks = tuple(banks)
    report_path, prediction_path, model_path = _fit_paths(context, stage, label, fold)
    if report_path.exists() and prediction_path.exists() and model_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if tuple(report["banks"]) != selected_banks:
            raise ValueError(f"checkpoint bank contract differs: {report_path}")
        return np.load(prediction_path), report

    fit_role, stop_role, score_role, stop_fold = role_rows(context, fold)
    fit_rows = sample_complete_series_rows(
        context, fit_role, train_cap, context.config.seed + 1009 * fold + len(selected_banks)
    )
    stop_rows = sample_complete_series_rows(
        context, stop_role, stop_cap, context.config.seed + 2003 * fold + len(selected_banks)
    )
    columns, names = selected_columns(context.cache.banks, selected_banks)
    metadata = context.source.data.metadata
    target = metadata["target"]
    online_time = metadata["time_online"]
    train_weight = pair_weights_from_reference(
        target[fit_rows], online_time[fit_rows], target[fit_role], online_time[fit_role]
    )
    stop_weight = pair_weights_from_reference(
        target[stop_rows], online_time[stop_rows], target[stop_role], online_time[stop_role]
    )
    print(
        f"{stage}/{label}/fold{fold}: {fit_rows.size:,} fit, {stop_rows.size:,} stop, "
        f"{score_role.size:,} score, {columns.size} features",
        flush=True,
    )
    started = time.perf_counter()
    x_train = matrix_rows(context.matrix, fit_rows, columns)
    x_stop = matrix_rows(context.matrix, stop_rows, columns)
    train_set = lgb.Dataset(x_train, label=target[fit_rows], weight=train_weight)
    stop_set = lgb.Dataset(
        x_stop, label=target[stop_rows], weight=stop_weight, reference=train_set
    )
    parameters = _lightgbm_parameters(
        context, fold=fold, bank_count=len(selected_banks)
    )

    def evaluator(prediction: np.ndarray, dataset: lgb.Dataset):
        return "tsroc", tsauc(dataset.get_label(), prediction, online_time[stop_rows]), True

    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=rounds,
        valid_sets=[stop_set],
        feval=evaluator,
        callbacks=[
            lgb.early_stopping(patience, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    prediction = predict_rows(
        booster,
        context.matrix,
        score_role,
        columns,
        context.config.prediction_chunk_rows,
    )
    score = float(tsauc(target[score_role], prediction, online_time[score_role]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction)
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    report = {
        "experiment_version": MODEL8_VERSION,
        "stage": stage,
        "label": label,
        "score_fold": fold,
        "stop_fold": stop_fold,
        "banks": list(selected_banks),
        "feature_count": int(columns.size),
        "feature_names_hash": _hash_json(names),
        "fit_rows": int(fit_rows.size),
        "fit_series": int(np.unique(metadata["series_position"][fit_rows]).size),
        "stop_rows": int(stop_rows.size),
        "stop_series": int(np.unique(metadata["series_position"][stop_rows]).size),
        "score_rows": int(score_role.size),
        "score_series": int(np.unique(metadata["series_position"][score_role]).size),
        "best_iteration": int(booster.best_iteration or rounds),
        "held_tsroc": score,
        "seconds": float(time.perf_counter() - started),
        "outer_holdout_read": False,
    }
    _atomic_json(report_path, report)
    print(
        f"{stage}/{label}/fold{fold}: TSAUC={score:.6f}, "
        f"best_iteration={report['best_iteration']}, {report['seconds'] / 60:.2f} min",
        flush=True,
    )
    del x_train, x_stop, train_set, stop_set, booster
    gc.collect()
    return prediction, report


def _assemble_oof(
    context: Model8Context,
    *,
    stage: str,
    label: str,
    banks_by_fold: Sequence[Sequence[str]],
    train_cap: int,
    stop_cap: int,
    rounds: int,
    patience: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(context.development_rows.size, np.nan, dtype=np.float32)
    reports = []
    for fold, banks in enumerate(banks_by_fold):
        prediction, report = fit_fold(
            context,
            stage=stage,
            label=label,
            banks=banks,
            fold=fold,
            train_cap=train_cap,
            stop_cap=stop_cap,
            rounds=rounds,
            patience=patience,
        )
        local = np.flatnonzero(context.local_fold == fold)
        if prediction.shape != (local.size,):
            raise ValueError("fold prediction has the wrong number of rows")
        output[local] = prediction
        reports.append(report)
    if not np.all(np.isfinite(output)):
        raise ValueError("OOF assembly left missing predictions")
    return output, reports


def _fold_scores(context: Model8Context, prediction: np.ndarray) -> list[float]:
    metadata = context.source.data.metadata
    target = metadata["target"][context.development_rows]
    online_time = metadata["time_online"][context.development_rows]
    return [
        float(tsauc(target[context.local_fold == fold], prediction[context.local_fold == fold], online_time[context.local_fold == fold]))
        for fold in range(context.config.folds)
    ]


def _prediction_report(context: Model8Context, prediction: np.ndarray) -> dict[str, Any]:
    scores = _fold_scores(context, prediction)
    metadata = context.source.data.metadata
    rows = context.development_rows
    report = _mean_se(scores)
    report.update(
        {
            "robust_fold_score": robust_score(scores, context.config.uncertainty_penalty),
            "pooled_development_oof": float(
                tsauc(metadata["target"][rows], prediction, metadata["time_online"][rows])
            ),
        }
    )
    return report


def nested_baseline_blend(
    context: Model8Context, candidate: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose one blend weight on four folds and apply it to the fifth."""

    output = np.empty_like(candidate, dtype=np.float32)
    recipes = []
    metadata = context.source.data.metadata
    rows = context.development_rows
    target = metadata["target"][rows]
    time_online = metadata["time_online"][rows]
    for held in range(context.config.folds):
        tune_folds = [fold for fold in range(context.config.folds) if fold != held]
        best = None
        for weight in context.config.blend_weights:
            prediction = _blend(context.baseline, candidate, weight)
            scores = [
                float(tsauc(target[context.local_fold == fold], prediction[context.local_fold == fold], time_online[context.local_fold == fold]))
                for fold in tune_folds
            ]
            criterion = robust_score(scores, context.config.uncertainty_penalty)
            key = (criterion, -weight)
            if best is None or key > best[0]:
                best = (key, float(weight), scores)
        assert best is not None
        selected = _blend(context.baseline, candidate, best[1])
        mask = context.local_fold == held
        output[mask] = selected[mask]
        recipes.append(
            {
                "held_fold": held,
                "candidate_weight": best[1],
                "tuning_fold_scores": best[2],
                "tuning_robust_score": best[0][0],
                "held_tsroc": float(tsauc(target[mask], output[mask], time_online[mask])),
            }
        )
    return output, {"method": "nested logit blend with Model 6 candidate OOF", "held_recipes": recipes}


def run_detector_screen(context: Model8Context) -> dict[str, Any]:
    """Fit all six banks and measure standalone plus baseline-incremental value."""

    output_path = context.config.artifact_path / "screen/summary.json"
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    baseline_report = _prediction_report(context, context.baseline)
    banks: dict[str, Any] = {}
    for bank in context.config.detector_banks:
        oof, fit_reports = _assemble_oof(
            context,
            stage="screen",
            label=bank,
            banks_by_fold=[(bank,)] * context.config.folds,
            train_cap=context.config.screen_train_rows,
            stop_cap=context.config.screen_stop_rows,
            rounds=context.config.screen_rounds,
            patience=context.config.screen_patience,
        )
        oof_path = context.config.artifact_path / f"screen/{bank}/oof.f32.npy"
        np.save(oof_path, oof)
        standalone = _prediction_report(context, oof)
        blended, blend_detail = nested_baseline_blend(context, oof)
        blend_report = _prediction_report(context, blended)
        baseline_scores = np.asarray(baseline_report["fold_scores"])
        deltas = np.asarray(blend_report["fold_scores"]) - baseline_scores
        delta_report = _mean_se(deltas)
        delta_robust = robust_score(deltas, context.config.uncertainty_penalty)
        selected = bool(
            standalone["robust_fold_score"] > context.config.detector_chance_threshold
            and delta_robust > 0.0
        )
        banks[bank] = {
            "feature_count": len(feature_names((bank,))),
            "standalone": standalone,
            "nested_baseline_blend": {**blend_detail, "score": blend_report},
            "increment_over_baseline": {**delta_report, "robust_gain": delta_robust},
            "selected_by_full_development_diagnostic": selected,
            "fit_reports": fit_reports,
            "oof_path": str(oof_path.relative_to(context.config.project_root)),
        }
        print(
            f"{bank}: standalone={standalone['fold_mean']:.6f}, "
            f"nested={blend_report['fold_mean']:.6f}, robust_delta={delta_robust:+.6f}, "
            f"selected={selected}",
            flush=True,
        )
    report = {
        "stage": "six_detector_screen",
        "selection_rule": (
            "standalone mean-0.5*SE > 0.5 and baseline-blend delta mean-0.5*SE > 0"
        ),
        "baseline": baseline_report,
        "banks": banks,
        "outer_holdout_read": False,
    }
    _atomic_json(output_path, report)
    return report


def _load_bank_predictions(context: Model8Context) -> dict[str, np.ndarray]:
    return {
        bank: np.load(context.config.artifact_path / f"screen/{bank}/oof.f32.npy")
        for bank in context.config.detector_banks
    }


def _select_banks_without_fold(
    context: Model8Context,
    bank_predictions: dict[str, np.ndarray],
    held: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Apply the detector rule using no labels from ``held``."""

    metadata = context.source.data.metadata
    rows = context.development_rows
    target = metadata["target"][rows]
    online_time = metadata["time_online"][rows]
    tune_folds = [fold for fold in range(context.config.folds) if fold != held]
    baseline_scores = [
        float(tsauc(target[context.local_fold == fold], context.baseline[context.local_fold == fold], online_time[context.local_fold == fold]))
        for fold in tune_folds
    ]
    selected = []
    details = {}
    for bank in context.config.detector_banks:
        candidate = bank_predictions[bank]
        standalone_scores = [
            float(tsauc(target[context.local_fold == fold], candidate[context.local_fold == fold], online_time[context.local_fold == fold]))
            for fold in tune_folds
        ]
        best = None
        for weight in context.config.blend_weights:
            prediction = _blend(context.baseline, candidate, weight)
            blend_scores = [
                float(tsauc(target[context.local_fold == fold], prediction[context.local_fold == fold], online_time[context.local_fold == fold]))
                for fold in tune_folds
            ]
            criterion = robust_score(blend_scores, context.config.uncertainty_penalty)
            key = (criterion, -weight)
            if best is None or key > best[0]:
                best = (key, float(weight), blend_scores)
        assert best is not None
        delta = np.asarray(best[2]) - np.asarray(baseline_scores)
        standalone_robust = robust_score(standalone_scores, context.config.uncertainty_penalty)
        delta_robust = robust_score(delta, context.config.uncertainty_penalty)
        keep = standalone_robust > context.config.detector_chance_threshold and delta_robust > 0
        if keep:
            selected.append(bank)
        details[bank] = {
            "standalone_tuning_scores": standalone_scores,
            "standalone_robust": standalone_robust,
            "best_candidate_weight": best[1],
            "blend_tuning_scores": best[2],
            "increment_tuning_scores": delta.tolist(),
            "increment_robust": delta_robust,
            "selected": keep,
        }
    return tuple(selected), {"held_fold": held, "tuning_folds": tune_folds, "banks": details}


def nested_greedy_bank_blend(
    context: Model8Context, bank_predictions: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Greedily add at most three bank OOF predictions without each held fold."""

    metadata = context.source.data.metadata
    rows = context.development_rows
    target = metadata["target"][rows]
    online_time = metadata["time_online"][rows]
    output = np.empty(context.development_rows.size, dtype=np.float32)
    held_recipes = []
    for held in range(context.config.folds):
        tune = [fold for fold in range(context.config.folds) if fold != held]
        current = np.asarray(context.baseline, dtype=np.float32)
        current_scores = [
            float(tsauc(target[context.local_fold == fold], current[context.local_fold == fold], online_time[context.local_fold == fold]))
            for fold in tune
        ]
        current_robust = robust_score(current_scores, context.config.uncertainty_penalty)
        unused = list(context.config.detector_banks)
        recipe = []
        for _ in range(context.config.greedy_max_new_banks):
            best = None
            for bank in unused:
                for weight in context.config.blend_weights:
                    prediction = _blend(current, bank_predictions[bank], weight)
                    scores = [
                        float(tsauc(target[context.local_fold == fold], prediction[context.local_fold == fold], online_time[context.local_fold == fold]))
                        for fold in tune
                    ]
                    criterion = robust_score(scores, context.config.uncertainty_penalty)
                    key = (criterion, -weight, bank)
                    if best is None or key > best[0]:
                        best = (key, bank, float(weight), scores, prediction)
            assert best is not None
            gain = best[0][0] - current_robust
            if gain <= context.config.greedy_min_robust_gain + 1e-12:
                break
            current = best[4]
            current_scores = best[3]
            current_robust = best[0][0]
            unused.remove(best[1])
            recipe.append(
                {
                    "bank": best[1],
                    "bank_weight": best[2],
                    "tuning_fold_scores": best[3],
                    "robust_gain": gain,
                }
            )
        mask = context.local_fold == held
        output[mask] = current[mask]
        held_recipes.append(
            {
                "held_fold": held,
                "tuning_folds": tune,
                "recipe": recipe,
                "held_tsroc": float(tsauc(target[mask], output[mask], online_time[mask])),
            }
        )
    return output, {"method": "nested greedy logit blend of cross-fitted bank models", "held_recipes": held_recipes}


def run_final_evaluation(context: Model8Context) -> dict[str, Any]:
    """Evaluate fixed all-bank and fold-nested selected-bank architectures."""

    output_path = context.config.artifact_path / "final/summary.json"
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    screen = run_detector_screen(context)
    del screen
    bank_predictions = _load_bank_predictions(context)

    all_oof, all_fit_reports = _assemble_oof(
        context,
        stage="combined",
        label="all_six",
        banks_by_fold=[context.config.detector_banks] * context.config.folds,
        train_cap=context.config.combined_train_rows,
        stop_cap=context.config.combined_stop_rows,
        rounds=context.config.combined_rounds,
        patience=context.config.combined_patience,
    )
    all_blended, all_blend_detail = nested_baseline_blend(context, all_oof)

    selected_by_fold = []
    selection_details = []
    for fold in range(context.config.folds):
        banks, detail = _select_banks_without_fold(context, bank_predictions, fold)
        selected_by_fold.append(banks)
        selection_details.append(detail)
    selected_oof = np.empty(context.development_rows.size, dtype=np.float32)
    selected_fit_reports = []
    for fold, banks in enumerate(selected_by_fold):
        local = np.flatnonzero(context.local_fold == fold)
        if not banks:
            selected_oof[local] = 0.5
            selected_fit_reports.append(
                {"score_fold": fold, "banks": [], "action": "no bank passed without held fold"}
            )
            continue
        prediction, fit_report = fit_fold(
            context,
            stage="combined",
            label="nested_selected",
            banks=banks,
            fold=fold,
            train_cap=context.config.combined_train_rows,
            stop_cap=context.config.combined_stop_rows,
            rounds=context.config.combined_rounds,
            patience=context.config.combined_patience,
        )
        selected_oof[local] = prediction
        selected_fit_reports.append(fit_report)

    greedy, greedy_detail = nested_greedy_bank_blend(context, bank_predictions)
    final_directory = context.config.artifact_path / "final"
    final_directory.mkdir(parents=True, exist_ok=True)
    np.save(final_directory / "all_six_oof.f32.npy", all_oof)
    np.save(final_directory / "all_six_nested_baseline_blend.f32.npy", all_blended)
    np.save(final_directory / "nested_selected_oof.f32.npy", selected_oof)
    np.save(final_directory / "nested_greedy_bank_blend.f32.npy", greedy)

    architectures = {
        "model6_candidate_baseline": _prediction_report(context, context.baseline),
        "all_six_raw": _prediction_report(context, all_oof),
        "all_six_nested_baseline_blend": _prediction_report(context, all_blended),
        "nested_selected_raw": _prediction_report(context, selected_oof),
        "nested_greedy_bank_blend": _prediction_report(context, greedy),
    }
    metadata = context.source.data.metadata
    rows = context.development_rows
    bootstrap = model4.cluster_bootstrap_tsauc(
        metadata["target"][rows],
        all_blended,
        metadata["time_online"][rows],
        metadata["series_position"][rows],
        replicates=context.config.bootstrap_replicates,
        seed=context.config.bootstrap_seed,
    )
    primary = architectures["all_six_nested_baseline_blend"]
    report = {
        "stage": "model8_nested_final",
        "primary_architecture": "all_six_nested_baseline_blend",
        "primary_was_predeclared": True,
        "architectures": architectures,
        "all_six_blend": all_blend_detail,
        "all_six_fit_reports": all_fit_reports,
        "selected_banks_without_each_fold": [list(banks) for banks in selected_by_fold],
        "selection_details": selection_details,
        "selected_fit_reports": selected_fit_reports,
        "greedy_bank_blend": greedy_detail,
        "primary_complete_series_bootstrap": bootstrap,
        "target_tsroc": context.config.target_tsroc,
        "target_reached_by_predeclared_primary": bool(
            primary["fold_mean"] >= context.config.target_tsroc
        ),
        "outer_holdout_read": False,
    }
    _atomic_json(output_path, report)
    return report


def compact_final(report: dict[str, Any]) -> dict[str, Any]:
    """Return the concise result printed by the CLI and notebook."""

    primary_name = report["primary_architecture"]
    primary = report["architectures"][primary_name]
    return {
        "stage": report["stage"],
        "primary_architecture": primary_name,
        "primary_fold_mean_tsroc": primary["fold_mean"],
        "primary_fold_standard_error": primary["fold_standard_error"],
        "primary_pooled_tsroc": primary["pooled_development_oof"],
        "baseline_fold_mean_tsroc": report["architectures"]["model6_candidate_baseline"]["fold_mean"],
        "nested_selected_fold_mean_tsroc": report["architectures"]["nested_selected_raw"]["fold_mean"],
        "greedy_fold_mean_tsroc": report["architectures"]["nested_greedy_bank_blend"]["fold_mean"],
        "bootstrap_standard_error": report["primary_complete_series_bootstrap"]["standard_error"],
        "bootstrap_95_interval": report["primary_complete_series_bootstrap"]["percentile_95_interval"],
        "target_reached": report["target_reached_by_predeclared_primary"],
        "outer_holdout_read": report["outer_holdout_read"],
    }


def outer_voter_roles(
    context: Model8Context, stop_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return development fit/stop rows and the untouched outer score rows.

    A deployment voter trains on four development folds, uses the fifth only
    for early stopping, and never includes an outer-holdout series in either
    role. The outer rows are returned only as unlabeled prediction inputs.
    """

    if stop_fold not in range(context.config.folds):
        raise ValueError("stop_fold is outside the Model 8 fold range")
    fit_rows = context.development_rows[context.local_fold != stop_fold]
    stop_rows = context.development_rows[context.local_fold == stop_fold]
    outer_rows = context.source.split.holdout_rows
    owner = context.source.data.metadata["series_position"]
    role_series = [set(np.unique(owner[rows])) for rows in (fit_rows, stop_rows, outer_rows)]
    if role_series[0] & role_series[1]:
        raise AssertionError("deployment fit and early-stop roles share series")
    if (role_series[0] | role_series[1]) & role_series[2]:
        raise AssertionError("outer-holdout series entered deployment fitting")
    return fit_rows, stop_rows, outer_rows


def select_deployment_weight(
    context: Model8Context, candidate: np.ndarray
) -> dict[str, Any]:
    """Freeze one all-development blend weight before outer inference."""

    candidate = np.asarray(candidate, dtype=np.float32)
    if candidate.shape != context.baseline.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("deployment candidate is not aligned to development OOF rows")
    rows = []
    best = None
    for weight in context.config.blend_weights:
        scores = _fold_scores(context, _blend(context.baseline, candidate, weight))
        criterion = robust_score(scores, context.config.uncertainty_penalty)
        row = {
            "candidate_weight": float(weight),
            "fold_scores": scores,
            "fold_mean": float(np.mean(scores)),
            "fold_standard_error": float(fold_standard_error(scores)),
            "robust_score": criterion,
        }
        rows.append(row)
        key = (criterion, -float(weight))
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return {
        "rule": "maximize development fold mean TSAUC - 0.5 * SE; prefer less new-model weight on ties",
        "uncertainty_penalty": context.config.uncertainty_penalty,
        "selected_candidate_weight": best[1]["candidate_weight"],
        "selected_development_fold_scores": best[1]["fold_scores"],
        "selected_development_robust_score": best[1]["robust_score"],
        "grid": rows,
    }


def freeze_outer_holdout(context: Model8Context) -> dict[str, Any]:
    """Hash every deployment choice before Model 8 reads outer predictions."""

    final_path = context.config.artifact_path / "final/summary.json"
    candidate_path = context.config.artifact_path / "final/all_six_oof.f32.npy"
    if not final_path.exists() or not candidate_path.exists():
        raise RuntimeError("finish the fixed Model 8 development experiment first")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final.get("primary_architecture") != "all_six_nested_baseline_blend":
        raise ValueError("Model 8 development primary differs from its fixed contract")
    candidate = np.load(candidate_path, allow_pickle=False)
    deployment_blend = select_deployment_weight(context, candidate)

    source_frozen_path = context.config.source_model6_path / "frozen_contract.json"
    if not source_frozen_path.exists():
        raise FileNotFoundError("Model 6 must have a frozen deployment contract")
    source_frozen = json.loads(source_frozen_path.read_text(encoding="utf-8"))
    voter_parameters = [
        _lightgbm_parameters(
            context,
            fold=fold,
            bank_count=len(context.config.detector_banks),
            seed_offset=80_000,
        )
        for fold in range(context.config.folds)
    ]
    contract = {
        "outer_protocol_version": "model8-outer-one-shot-v1",
        "experiment_version": MODEL8_VERSION,
        "source_experiment_manifest_sha256": _hash_file(
            context.config.artifact_path / "experiment_manifest.json"
        ),
        "source_development_final_sha256": _hash_file(final_path),
        "source_all_six_oof_sha256": _hash_file(candidate_path),
        "model6_frozen_contract_hash": source_frozen["contract_hash"],
        "primary_architecture": "five-voter all-six LightGBM + frozen Model 6 candidate blend",
        "detector_banks": list(context.config.detector_banks),
        "feature_signature": _feature_signature(context.config.detector_banks),
        "deployment_blend": deployment_blend,
        "new_model_voters": {
            "count": context.config.folds,
            "roles": "for voter f: early stop on development fold f; fit on the other four development folds",
            "aggregation": "arithmetic mean of five probability logits",
            "train_row_cap": context.config.combined_train_rows,
            "stop_row_cap": context.config.combined_stop_rows,
            "rounds": context.config.combined_rounds,
            "patience": context.config.combined_patience,
            "parameters": voter_parameters,
        },
        "outer_series": int(context.source.split.holdout_series_mask.sum()),
        "outer_rows": int(context.source.split.holdout_rows.size),
        "bootstrap_replicates": context.config.bootstrap_replicates,
        "bootstrap_seed": context.config.bootstrap_seed + 8,
        "model8_outer_labels_used_for_contract": False,
        "repository_holdout_previously_scored_by_model6": (
            context.config.source_model6_path / "holdout_results.json"
        ).exists(),
    }
    contract["contract_hash"] = _hash_json(contract)
    _write_or_validate(
        context.config.artifact_path / "outer_holdout/frozen_contract.json", contract
    )
    return contract


def _validate_outer_contract(
    context: Model8Context, frozen: dict[str, Any]
) -> None:
    path = context.config.artifact_path / "outer_holdout/frozen_contract.json"
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("supplied Model 8 outer contract differs from disk")
    unhashed = dict(frozen)
    claimed = unhashed.pop("contract_hash")
    if claimed != _hash_json(unhashed):
        raise ValueError("Model 8 outer contract hash is invalid")


def _model6_candidate_outer_prediction(
    context: Model8Context, frozen: dict[str, Any]
) -> np.ndarray:
    """Reconstruct frozen Model 6 candidate inference without outer labels."""

    directory = context.config.artifact_path / "outer_holdout/model6_candidate"
    prediction_path = directory / "prediction.f32.npy"
    manifest_path = directory / "manifest.json"
    if prediction_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model8_outer_contract_hash") != frozen["contract_hash"]:
            raise ValueError("cached Model 6 candidate prediction has another contract")
        prediction = np.load(prediction_path, allow_pickle=False)
        if prediction.shape != (context.source.split.holdout_rows.size,):
            raise ValueError("cached Model 6 candidate prediction has the wrong shape")
        return prediction.astype(np.float32, copy=False)

    source_frozen = model6.freeze_deployment(context.source)
    if source_frozen["contract_hash"] != frozen["model6_frozen_contract_hash"]:
        raise ValueError("Model 6 deployment contract changed after Model 8 freeze")
    family = "candidate"
    components = model6._holdout_stack_components(  # noqa: SLF001
        context.source, family, source_frozen["detector_stack"][family]
    )
    components.update(
        model6._holdout_feature_components(  # noqa: SLF001
            context.source, family, source_frozen["feature_finalists"][family]
        )
    )
    margins = {
        name: model6._safe_logit(value)  # noqa: SLF001
        for name, value in components.items()
    }
    family_report = source_frozen["family_ensembles"][family]
    raw = model6._apply_recipe(  # noqa: SLF001
        margins, family_report["deployment_recipe"]
    )
    owner = context.source.data.metadata["series_position"][
        context.source.split.holdout_rows
    ]
    prediction = model6._deployment_transform(  # noqa: SLF001
        raw, owner, family_report
    ).astype(np.float32)
    if prediction.shape != (context.source.split.holdout_rows.size,) or not np.all(
        np.isfinite(prediction)
    ):
        raise ValueError("Model 6 candidate outer prediction is invalid")
    directory.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction)
    _atomic_json(
        manifest_path,
        {
            "model8_outer_contract_hash": frozen["contract_hash"],
            "model6_frozen_contract_hash": source_frozen["contract_hash"],
            "prediction_sha256": _hash_file(prediction_path),
            "outer_rows": int(prediction.size),
            "outer_labels_read": False,
        },
    )
    return prediction


def _outer_voter_paths(
    context: Model8Context, stop_fold: int
) -> tuple[Path, Path, Path]:
    directory = context.config.artifact_path / "outer_holdout/all_six_voters"
    return (
        directory / f"voter_{stop_fold}.json",
        directory / f"voter_{stop_fold}.prediction.f32.npy",
        directory / f"voter_{stop_fold}.txt",
    )


def fit_outer_voter(
    context: Model8Context, frozen: dict[str, Any], stop_fold: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one development-only voter and predict unlabeled outer rows."""

    if context.cache is None or context.matrix is None:
        raise ValueError("outer voter fitting requires the Model 8 cache")
    _validate_outer_contract(context, frozen)
    report_path, prediction_path, model_path = _outer_voter_paths(context, stop_fold)
    if report_path.exists() and prediction_path.exists() and model_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("outer_contract_hash") != frozen["contract_hash"]:
            raise ValueError("cached outer voter belongs to another contract")
        return np.load(prediction_path, allow_pickle=False), report

    fit_role, stop_role, outer_rows = outer_voter_roles(context, stop_fold)
    fit_rows = sample_complete_series_rows(
        context,
        fit_role,
        context.config.combined_train_rows,
        context.config.seed + 81_000 + 1009 * stop_fold,
    )
    stop_rows = sample_complete_series_rows(
        context,
        stop_role,
        context.config.combined_stop_rows,
        context.config.seed + 82_000 + 2003 * stop_fold,
    )
    columns, names = selected_columns(
        context.cache.banks, context.config.detector_banks
    )
    metadata = context.source.data.metadata
    target = metadata["target"]
    online_time = metadata["time_online"]
    train_weight = pair_weights_from_reference(
        target[fit_rows], online_time[fit_rows], target[fit_role], online_time[fit_role]
    )
    stop_weight = pair_weights_from_reference(
        target[stop_rows],
        online_time[stop_rows],
        target[stop_role],
        online_time[stop_role],
    )
    print(
        f"outer/all_six/voter{stop_fold}: {fit_rows.size:,} fit, "
        f"{stop_rows.size:,} stop, {outer_rows.size:,} predict",
        flush=True,
    )
    started = time.perf_counter()
    x_train = matrix_rows(context.matrix, fit_rows, columns)
    x_stop = matrix_rows(context.matrix, stop_rows, columns)
    train_set = lgb.Dataset(x_train, label=target[fit_rows], weight=train_weight)
    stop_set = lgb.Dataset(
        x_stop, label=target[stop_rows], weight=stop_weight, reference=train_set
    )
    parameters = _lightgbm_parameters(
        context,
        fold=stop_fold,
        bank_count=len(context.config.detector_banks),
        seed_offset=80_000,
    )

    def evaluator(prediction: np.ndarray, dataset: lgb.Dataset):
        return "tsroc", tsauc(dataset.get_label(), prediction, online_time[stop_rows]), True

    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=context.config.combined_rounds,
        valid_sets=[stop_set],
        feval=evaluator,
        callbacks=[
            lgb.early_stopping(
                context.config.combined_patience,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    prediction = predict_rows(
        booster,
        context.matrix,
        outer_rows,
        columns,
        context.config.prediction_chunk_rows,
    ).astype(np.float32)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("outer voter produced non-finite predictions")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction)
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    report = {
        "outer_contract_hash": frozen["contract_hash"],
        "stop_fold": stop_fold,
        "banks": list(context.config.detector_banks),
        "feature_count": int(columns.size),
        "feature_names_hash": _hash_json(names),
        "fit_rows": int(fit_rows.size),
        "fit_series": int(np.unique(metadata["series_position"][fit_rows]).size),
        "stop_rows": int(stop_rows.size),
        "stop_series": int(np.unique(metadata["series_position"][stop_rows]).size),
        "best_iteration": int(booster.best_iteration or context.config.combined_rounds),
        "stop_tsroc": float(
            tsauc(
                target[stop_rows],
                booster.predict(x_stop, num_iteration=booster.best_iteration),
                online_time[stop_rows],
            )
        ),
        "outer_rows_predicted": int(outer_rows.size),
        "outer_labels_read": False,
        "seconds": float(time.perf_counter() - started),
    }
    _atomic_json(report_path, report)
    print(
        f"outer/all_six/voter{stop_fold}: stop TSAUC={report['stop_tsroc']:.6f}, "
        f"best_iteration={report['best_iteration']}, {report['seconds'] / 60:.2f} min",
        flush=True,
    )
    del x_train, x_stop, train_set, stop_set, booster
    gc.collect()
    return prediction, report


def build_frozen_outer_prediction(
    context: Model8Context, frozen: dict[str, Any]
) -> dict[str, Any]:
    """Build frozen outer predictions without indexing outer labels."""

    _validate_outer_contract(context, frozen)
    directory = context.config.artifact_path / "outer_holdout/predictions"
    manifest_path = directory / "manifest.json"
    output_paths = {
        "model6_candidate_baseline": directory / "model6_candidate.f32.npy",
        "all_six_raw": directory / "all_six_raw.f32.npy",
        "primary": directory / "primary.f32.npy",
    }
    if manifest_path.exists() and all(path.exists() for path in output_paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("outer_contract_hash") != frozen["contract_hash"]:
            raise ValueError("outer prediction manifest belongs to another contract")
        return manifest

    baseline = _model6_candidate_outer_prediction(context, frozen)
    voter_predictions = []
    voter_reports = []
    for stop_fold in range(context.config.folds):
        prediction, report = fit_outer_voter(context, frozen, stop_fold)
        voter_predictions.append(prediction)
        voter_reports.append(report)
    voter_margin = np.column_stack([_safe_logit(value) for value in voter_predictions])
    all_six = _logistic(voter_margin.mean(axis=1)).astype(np.float32)
    weight = float(frozen["deployment_blend"]["selected_candidate_weight"])
    primary = _blend(baseline, all_six, weight)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "model6_candidate_baseline": baseline,
        "all_six_raw": all_six,
        "primary": primary,
    }
    for name, prediction in arrays.items():
        np.save(output_paths[name], prediction.astype(np.float32))
    manifest = {
        "outer_contract_hash": frozen["contract_hash"],
        "outer_rows": int(primary.size),
        "selected_new_model_weight": weight,
        "voter_reports": voter_reports,
        "predictions": {
            name: str(path.relative_to(context.config.project_root))
            for name, path in output_paths.items()
        },
        "prediction_sha256": {
            name: _hash_file(path) for name, path in output_paths.items()
        },
        "outer_labels_read": False,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def evaluate_outer_holdout(
    context: Model8Context,
    frozen: dict[str, Any],
    prediction_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Open outer labels once after predictions and choices are frozen."""

    _validate_outer_contract(context, frozen)
    result_path = context.config.artifact_path / "outer_holdout/results.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("outer_contract_hash") != frozen["contract_hash"]:
            raise ValueError("saved outer result belongs to another contract")
        return result
    if prediction_manifest.get("outer_contract_hash") != frozen["contract_hash"]:
        raise ValueError("outer prediction manifest and frozen contract differ")
    predictions = {
        name: np.load(
            context.config.project_root / relative_path, allow_pickle=False
        )
        for name, relative_path in prediction_manifest["predictions"].items()
    }
    outer_rows = context.source.split.holdout_rows
    if any(value.shape != (outer_rows.size,) for value in predictions.values()):
        raise ValueError("outer prediction has the wrong number of rows")

    # This is deliberately the only Model 8 function that indexes outer labels.
    metadata = context.source.data.metadata
    target = metadata["target"][outer_rows]
    online_time = metadata["time_online"][outer_rows]
    owner = metadata["series_position"][outer_rows]
    scores = {
        name: float(tsauc(target, prediction, online_time))
        for name, prediction in predictions.items()
    }
    bootstrap = model4.cluster_bootstrap_tsauc(
        target,
        predictions["primary"],
        online_time,
        owner,
        replicates=int(frozen["bootstrap_replicates"]),
        seed=int(frozen["bootstrap_seed"]),
    )
    development = json.loads(
        (context.config.artifact_path / "final/summary.json").read_text(
            encoding="utf-8"
        )
    )
    result = {
        "stage": "model8_outer_holdout_one_shot",
        "outer_contract_hash": frozen["contract_hash"],
        "outer_holdout_accessed": True,
        "outer_series": int(context.source.split.holdout_series_mask.sum()),
        "outer_rows": int(outer_rows.size),
        "selected_new_model_weight": prediction_manifest[
            "selected_new_model_weight"
        ],
        "primary_outer_tsroc": scores["primary"],
        "model6_candidate_baseline_outer_tsroc": scores[
            "model6_candidate_baseline"
        ],
        "all_six_raw_outer_tsroc_diagnostic": scores["all_six_raw"],
        "primary_minus_baseline": scores["primary"]
        - scores["model6_candidate_baseline"],
        "primary_complete_series_bootstrap": bootstrap,
        "development_primary_fold_mean": development["architectures"][
            development["primary_architecture"]
        ]["fold_mean"],
        "target_tsroc": context.config.target_tsroc,
        "target_reached": bool(scores["primary"] >= context.config.target_tsroc),
        "selection_or_weight_changed_after_outer_score": False,
        "honesty_note": (
            "This is a procedurally frozen Model 8 test, but the same repository "
            "outer split was previously opened by Model 6. Treat it as a useful "
            "comparison, not a pristine estimate of competition generalization."
        ),
    }
    _atomic_json(result_path, result)
    return result


def compact_outer(result: dict[str, Any]) -> dict[str, Any]:
    """Return the concise one-shot outer result for the CLI and notebook."""

    bootstrap = result["primary_complete_series_bootstrap"]
    return {
        "stage": result["stage"],
        "primary_outer_tsroc": result["primary_outer_tsroc"],
        "model6_candidate_baseline_outer_tsroc": result[
            "model6_candidate_baseline_outer_tsroc"
        ],
        "primary_minus_baseline": result["primary_minus_baseline"],
        "all_six_raw_outer_tsroc_diagnostic": result[
            "all_six_raw_outer_tsroc_diagnostic"
        ],
        "selected_new_model_weight": result["selected_new_model_weight"],
        "bootstrap_standard_error": bootstrap["standard_error"],
        "bootstrap_95_interval": bootstrap["percentile_95_interval"],
        "target_reached": result["target_reached"],
        "outer_holdout_accessed": result["outer_holdout_accessed"],
        "honesty_note": result["honesty_note"],
    }
