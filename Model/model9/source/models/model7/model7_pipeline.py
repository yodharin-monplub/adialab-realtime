"""Model 7: exact-time pairwise ranking with repeated, series-grouped CV.

The source feature cache and detector selection come from Model 6, but Model
7 never indexes the 1,334 series in Model 6's revealed outer holdout.  Three
new five-fold assignments are made solely inside the old development set.
For every scored fold, feature ranking, complete-series row sampling, model
fitting, early stopping, ensemble selection, and sequence-filter selection
exclude that fold.
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
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from models.model4 import model4_pipeline as base
from models.model6.config import make_model6_config
from models.model6.model6_pipeline import load_context as load_model6_context
from models.model7.config import MODEL7_VERSION, Model7Config
from train_cv import make_series_folds, matrix_rows, pair_weights_from_reference
from tsauc import fold_standard_error, tsauc


@dataclass(frozen=True)
class ModelSpec:
    """One predeclared learner; no CV result changes this specification."""

    name: str
    kind: str
    train_rows: int


@dataclass
class Model7Context:
    """Train-derived matrices plus repeated folds restricted to development."""

    config: Model7Config
    source: Any
    development_rows: np.ndarray
    repeat_series_fold: np.ndarray
    repeat_row_fold: np.ndarray


def model_specs(config: Model7Config) -> tuple[ModelSpec, ...]:
    """Return the fixed ranking/baseline candidates in stable order."""

    return (
        ModelSpec("xgb_pair_d3_800k", "xgb_pairwise", config.xgb_small_train_rows),
        ModelSpec("xgb_pair_d3_1200k", "xgb_pairwise", config.large_train_rows),
        ModelSpec("goss_binary_1200k", "lightgbm_goss", config.large_train_rows),
    )


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


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_ready_json(value), indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_or_validate(path: Path, value: Any) -> None:
    ready = _ready_json(value)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != ready:
            raise ValueError(
                f"{path} belongs to a different Model 7 protocol; change the "
                "configuration so it receives a new run directory"
            )
        return
    _atomic_json(path, ready)


def _mean_se(scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite and non-empty")
    return {
        "scores": values.tolist(),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else None,
        "standard_error": (
            float(fold_standard_error(values)) if values.size > 1 else None
        ),
    }


def robust_score(scores: Sequence[float], penalty: float) -> float:
    """Return mean minus a declared multiple of fold standard error."""

    summary = _mean_se(scores)
    se = summary["standard_error"]
    return float(summary["mean"] - penalty * (0.0 if se is None else se))


def _sanitize(matrix: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(matrix, dtype=np.float32)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return output


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(bounded) - np.log1p(-bounded)


def _logistic(margin: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(margin, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def _cap_rows(rows: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size <= maximum:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=maximum, replace=False))


def load_context(config: Model7Config) -> Model7Context:
    """Load source caches and create repeats without Model 6 holdout series."""

    config.validate()
    source = load_model6_context(make_model6_config(config.project_root))
    development_series = np.asarray(
        source.split.development_series_mask, dtype=bool
    )
    prohibited_series = np.asarray(source.split.holdout_series_mask, dtype=bool)
    if np.any(development_series & prohibited_series):
        raise AssertionError("source development and revealed holdout overlap")
    development_positions = np.flatnonzero(development_series)
    development_rows = np.flatnonzero(source.split.development_row_mask).astype(
        np.int64
    )
    series_folds = np.full(
        (len(config.repeat_seeds), development_series.size), -1, dtype=np.int8
    )
    for repeat, seed in enumerate(config.repeat_seeds):
        local = make_series_folds(
            source.data.metadata["online_lengths"][development_positions],
            source.data.metadata["tau_index"][development_positions],
            n_splits=config.folds_per_repeat,
            seed=seed,
        ).astype(np.int8)
        series_folds[repeat, development_positions] = local
    if np.any(series_folds[:, prohibited_series] != -1):
        raise AssertionError("a revealed holdout series entered Model 7")
    if any(
        set(np.unique(series_folds[repeat, development_positions]))
        != set(range(config.folds_per_repeat))
        for repeat in range(len(config.repeat_seeds))
    ):
        raise AssertionError("a repeated development split is incomplete")
    series_position = source.data.metadata["series_position"][development_rows]
    row_folds = np.stack(
        [fold[series_position] for fold in series_folds], axis=0
    ).astype(np.int8)
    split_path = config.artifact_path / "repeated_splits.npz"
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["series_folds"], series_folds):
                raise ValueError("saved Model 7 repeated splits differ")
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, series_folds=series_folds)

    manifest = {
        "experiment_version": MODEL7_VERSION,
        "protocol": config.protocol(),
        "protocol_hash": _hash_json(config.identity_protocol()),
        "source_model6_run": source.config.artifact_directory,
        "source_detector_selection_hash": source.source_selection["selection_hash"],
        "allowed_population": "Model 6 development series only",
        "development_series": int(development_series.sum()),
        "development_rows": int(development_rows.size),
        "prohibited_revealed_holdout_series": int(prohibited_series.sum()),
        "prohibited_series_mask_hash": _array_hash(prohibited_series.astype(np.uint8)),
        "model6_holdout_results_read": False,
        "model6_holdout_predictions_read": False,
        "repeat_fold_series_counts": {
            str(repeat): [
                int(np.sum(series_folds[repeat] == fold))
                for fold in range(config.folds_per_repeat)
            ]
            for repeat in range(len(config.repeat_seeds))
        },
        "models": [spec.__dict__ for spec in model_specs(config)],
        "selection_rule": "mean TSAUC - uncertainty_penalty * fold SE",
    }
    _write_or_validate(config.artifact_path / "experiment_manifest.json", manifest)
    return Model7Context(
        config=config,
        source=source,
        development_rows=development_rows,
        repeat_series_fold=series_folds,
        repeat_row_fold=row_folds,
    )


def context_summary(context: Model7Context) -> dict[str, Any]:
    return {
        "stage": "model7_status",
        "run": context.config.artifact_directory,
        "development_series": int(
            context.source.split.development_series_mask.sum()
        ),
        "development_rows": int(context.development_rows.size),
        "prohibited_holdout_series": int(
            context.source.split.holdout_series_mask.sum()
        ),
        "repeat_seeds": list(context.config.repeat_seeds),
        "models": [spec.name for spec in model_specs(context.config)],
        "device": context.source.actual_device,
    }


def role_rows(
    context: Model7Context, repeat: int, score_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return full-series train, stop, and score rows for one repeated fold."""

    config = context.config
    if repeat not in range(len(config.repeat_seeds)):
        raise ValueError("repeat is outside the configured range")
    if score_fold not in range(config.folds_per_repeat):
        raise ValueError("score_fold is outside the configured range")
    stop_fold = (score_fold + 1) % config.folds_per_repeat
    local_fold = context.repeat_row_fold[repeat]
    train_local = (local_fold != score_fold) & (local_fold != stop_fold)
    stop_local = local_fold == stop_fold
    score_local = local_fold == score_fold
    train = context.development_rows[train_local]
    stop = context.development_rows[stop_local]
    score = context.development_rows[score_local]
    prohibited = context.source.split.holdout_row_mask
    if np.any(prohibited[train]) or np.any(prohibited[stop]) or np.any(prohibited[score]):
        raise AssertionError("Model 6 holdout row entered a Model 7 role")
    series_position = context.source.data.metadata["series_position"]
    role_series = [set(np.unique(series_position[rows])) for rows in (train, stop, score)]
    if role_series[0] & role_series[1] or role_series[0] & role_series[2]:
        raise AssertionError("train series overlap another role")
    if role_series[1] & role_series[2]:
        raise AssertionError("stop and score series overlap")
    return train, stop, score, stop_fold


def sample_complete_series_rows(
    context: Model7Context, rows: np.ndarray, maximum_rows: int, seed: int
) -> np.ndarray:
    """Cap a role by deterministic complete-series sampling."""

    rows = np.asarray(rows, dtype=np.int64)
    if rows.size <= maximum_rows:
        return rows
    metadata = context.source.data.metadata
    row_series = metadata["series_position"][rows]
    series = np.unique(row_series)
    rng = np.random.default_rng(seed)
    shuffled = series.copy()
    rng.shuffle(shuffled)
    lengths = metadata["online_lengths"][shuffled]
    cumulative = np.cumsum(lengths, dtype=np.int64)
    count = int(np.searchsorted(cumulative, maximum_rows, side="right"))
    count = max(1, count)
    chosen = np.sort(shuffled[:count])
    selected = rows[np.isin(row_series, chosen)]
    if not np.array_equal(
        np.unique(metadata["series_position"][selected]), chosen
    ):
        raise AssertionError("complete-series sampling lost a chosen series")
    return selected


def _ranking_paths(context: Model7Context, repeat: int, fold: int) -> tuple[Path, Path]:
    directory = context.config.artifact_path / "rankings" / f"repeat_{repeat}"
    return directory / f"fold_{fold}.npz", directory / f"fold_{fold}.json"


def build_fisher_ranking(
    context: Model7Context, repeat: int, fold: int
) -> tuple[np.ndarray, list[str]]:
    """Select Fisher columns using only this fold's fitting role."""

    output, report_path = _ranking_paths(context, repeat, fold)
    train_role, _, _, stop_fold = role_rows(context, repeat, fold)
    source_columns = context.source.family_columns[context.config.feature_family]
    source_names = context.source.family_names[context.config.feature_family]
    contract = {
        "experiment_version": MODEL7_VERSION,
        "repeat": repeat,
        "score_fold": fold,
        "stop_fold": stop_fold,
        "train_series_hash": _array_hash(
            np.unique(
                context.source.data.metadata["series_position"][train_role]
            ).astype(np.int32)
        ),
        "method": "weighted Fisher class separation",
        "feature_family": context.config.feature_family,
        "source_feature_count": int(source_columns.size),
        "selected_feature_count": context.config.feature_count,
        "ranking_sample_rows": context.config.ranking_sample_rows,
        "score_and_stop_labels_excluded": True,
        "model6_holdout_excluded": True,
    }
    contract_hash = _hash_json(contract)
    if output.exists() and report_path.exists():
        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        if saved_report.get("contract_hash") != contract_hash:
            raise ValueError(f"stale Fisher ranking: {output}")
        with np.load(output, allow_pickle=False) as saved:
            columns = saved["columns"].astype(np.int32)
            names = saved["names"].astype(str).tolist()
        return columns, names

    rows = _cap_rows(
        train_role,
        context.config.ranking_sample_rows,
        context.config.seed + 10_000 * repeat + 101 * fold,
    )
    metadata = context.source.data.metadata
    y = metadata["target"][rows]
    weights = pair_weights_from_reference(
        y,
        metadata["time_online"][rows],
        metadata["target"][train_role],
        metadata["time_online"][train_role],
    ).astype(np.float64)
    fisher = np.zeros(source_columns.size, dtype=np.float64)
    for left in range(0, source_columns.size, 256):
        right = min(left + 256, source_columns.size)
        matrix = _sanitize(
            matrix_rows(
                context.source.data.joint_matrix,
                rows,
                source_columns[left:right],
            )
        ).astype(np.float64)
        moments: list[tuple[np.ndarray, np.ndarray]] = []
        for label in (0, 1):
            selected = y == label
            class_weight = weights[selected]
            denominator = max(float(class_weight.sum()), 1e-12)
            mean = np.sum(
                matrix[selected] * class_weight[:, None], axis=0
            ) / denominator
            variance = np.sum(
                (matrix[selected] - mean) ** 2 * class_weight[:, None], axis=0
            ) / denominator
            moments.append((mean, variance))
        fisher[left:right] = np.abs(moments[1][0] - moments[0][0]) / np.sqrt(
            moments[1][1] + moments[0][1] + 1e-8
        )
        del matrix
    fisher = np.nan_to_num(fisher, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.lexsort((np.arange(fisher.size), -fisher)).astype(np.int32)
    local = order[: context.config.feature_count]
    columns = source_columns[local].astype(np.int32)
    names = [source_names[int(index)] for index in local]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        columns=columns,
        names=np.asarray(names, dtype=np.str_),
        local_order=order,
        fisher=fisher.astype(np.float32),
    )
    _atomic_json(
        report_path,
        {
            **contract,
            "ranking_rows": int(rows.size),
            "column_hash": _array_hash(columns),
            "contract_hash": contract_hash,
        },
    )
    print(f"ranking repeat {repeat} fold {fold}: complete", flush=True)
    return columns, names


def query_group_layout(
    target: np.ndarray, time_online: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-group row mask and official pair weights per time group."""

    y = np.asarray(target, dtype=np.int8)
    time_value = np.asarray(time_online, dtype=np.int64)
    if y.ndim != 1 or y.shape != time_value.shape:
        raise ValueError("target and time_online must be same-length vectors")
    if time_value.size and np.any(time_value[1:] < time_value[:-1]):
        raise ValueError("ranking rows must be sorted by time_online")
    if time_value.size == 0:
        return np.empty(0, dtype=bool), np.empty(0, dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, time_value[1:] != time_value[:-1]])
    ends = np.r_[starts[1:], y.size]
    counts = ends - starts
    positives = np.add.reduceat(y.astype(np.int64), starts)
    negatives = counts - positives
    pairs = positives * negatives
    valid_groups = pairs > 0
    keep = np.repeat(valid_groups, counts)
    weights = pairs[valid_groups].astype(np.float64)
    if weights.size:
        weights /= weights.mean()
    return keep, weights.astype(np.float32)


def _query_rows(
    context: Model7Context, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = context.source.data.metadata
    order = np.lexsort(
        (
            metadata["series_position"][rows],
            metadata["time_online"][rows],
        )
    )
    sorted_rows = np.asarray(rows, dtype=np.int64)[order]
    target = metadata["target"][sorted_rows]
    time_online = metadata["time_online"][sorted_rows]
    keep, group_weight = query_group_layout(target, time_online)
    return (
        sorted_rows[keep],
        target[keep],
        time_online[keep],
        group_weight,
    )


def _calibration(raw_margin: np.ndarray) -> tuple[float, float]:
    margin = np.asarray(raw_margin, dtype=np.float64)
    center = float(np.median(margin))
    q25, q75 = np.percentile(margin, [25.0, 75.0])
    scale = max(float((q75 - q25) / 1.349), float(margin.std()), 1e-3)
    return center, scale


def _calibrated_probability(
    raw_margin: np.ndarray, center: float, scale: float
) -> np.ndarray:
    return _logistic((np.asarray(raw_margin, dtype=np.float64) - center) / scale)


def _predict_xgb_rows(
    context: Model7Context,
    booster: Any,
    rows: np.ndarray,
    columns: np.ndarray,
    best_iteration: int,
    center: float,
    scale: float,
) -> np.ndarray:
    import xgboost as xgb

    output = np.empty(rows.size, dtype=np.float32)
    chunk = context.config.prediction_chunk_rows
    for left in range(0, rows.size, chunk):
        right = min(left + chunk, rows.size)
        matrix = _sanitize(
            matrix_rows(
                context.source.data.joint_matrix, rows[left:right], columns
            )
        )
        raw = booster.predict(
            xgb.DMatrix(matrix), iteration_range=(0, best_iteration + 1)
        )
        output[left:right] = _calibrated_probability(raw, center, scale).astype(
            np.float32
        )
        del matrix
    return output


def _predict_lgb_rows(
    context: Model7Context,
    booster: lgb.Booster,
    rows: np.ndarray,
    columns: np.ndarray,
    best_iteration: int,
    center: float,
    scale: float,
) -> np.ndarray:
    output = np.empty(rows.size, dtype=np.float32)
    chunk = context.config.prediction_chunk_rows
    for left in range(0, rows.size, chunk):
        right = min(left + chunk, rows.size)
        matrix = _sanitize(
            matrix_rows(
                context.source.data.joint_matrix, rows[left:right], columns
            )
        )
        probability = booster.predict(matrix, num_iteration=best_iteration)
        output[left:right] = _calibrated_probability(
            _safe_logit(probability), center, scale
        ).astype(np.float32)
        del matrix
    return output


def _fit_path(context: Model7Context, spec: ModelSpec, repeat: int, fold: int) -> Path:
    return (
        context.config.artifact_path
        / "fits"
        / spec.name
        / f"repeat_{repeat}"
        / f"fold_{fold}.json"
    )


def fit_fold(
    context: Model7Context, spec: ModelSpec, repeat: int, fold: int
) -> dict[str, Any]:
    """Fit one fixed model with disjoint complete-series train/stop/score roles."""

    config = context.config
    result_path = _fit_path(context, spec, repeat, fold)
    prediction_path = result_path.with_suffix(".prediction.f32.npy")
    columns_path = result_path.with_suffix(".columns.npz")
    model_suffix = ".ubj" if spec.kind == "xgb_pairwise" else ".txt"
    model_path = result_path.with_suffix(model_suffix)
    train_role, stop_role, score_rows, stop_fold = role_rows(context, repeat, fold)
    columns, names = build_fisher_ranking(context, repeat, fold)
    name_seed = int(hashlib.sha256(spec.name.encode()).hexdigest()[:8], 16)
    stage_seed = int(config.seed + 100_000 * repeat + 10_000 * fold + name_seed % 9_973)
    train_rows = sample_complete_series_rows(
        context, train_role, spec.train_rows, stage_seed + 1
    )
    stop_rows = sample_complete_series_rows(
        context, stop_role, config.stop_rows, stage_seed + 2
    )
    contract = {
        "experiment_version": MODEL7_VERSION,
        "model": spec.__dict__,
        "repeat": repeat,
        "repeat_seed": config.repeat_seeds[repeat],
        "score_fold": fold,
        "stop_fold": stop_fold,
        "feature_method": config.ranking_method,
        "feature_count": int(columns.size),
        "column_hash": _array_hash(columns),
        "train_series_hash": _array_hash(
            np.unique(
                context.source.data.metadata["series_position"][train_rows]
            ).astype(np.int32)
        ),
        "stop_series_hash": _array_hash(
            np.unique(
                context.source.data.metadata["series_position"][stop_rows]
            ).astype(np.int32)
        ),
        "score_series_hash": _array_hash(
            np.unique(
                context.source.data.metadata["series_position"][score_rows]
            ).astype(np.int32)
        ),
        "rounds": config.boosting_rounds,
        "patience": config.early_stopping_rounds,
        "seed": stage_seed,
        "model6_holdout_excluded": True,
    }
    contract_hash = _hash_json(contract)
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("contract_hash") != contract_hash:
            raise ValueError(f"stale Model 7 fit checkpoint: {result_path}")
        if not prediction_path.exists() or not columns_path.exists() or not model_path.exists():
            raise ValueError(f"incomplete Model 7 fit checkpoint: {result_path}")
        print(
            f"{spec.name}/repeat{repeat}/fold{fold}: checkpoint "
            f"TSAUC={saved['tsauc']:.6f}",
            flush=True,
        )
        return saved

    result_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"{spec.name}/repeat{repeat}/fold{fold}: {train_rows.size:,} train, "
        f"{stop_rows.size:,} stop, {score_rows.size:,} score, "
        f"{columns.size} Fisher features",
        flush=True,
    )
    started = time.perf_counter()
    metadata = context.source.data.metadata
    if spec.kind == "xgb_pairwise":
        import xgboost as xgb

        query_train, y_train, time_train, train_group_weight = _query_rows(
            context, train_rows
        )
        query_stop, y_stop, time_stop, stop_group_weight = _query_rows(
            context, stop_rows
        )
        x_train = _sanitize(
            matrix_rows(context.source.data.joint_matrix, query_train, columns)
        )
        x_stop = _sanitize(
            matrix_rows(context.source.data.joint_matrix, query_stop, columns)
        )
        dtrain = xgb.QuantileDMatrix(
            x_train,
            label=y_train,
            qid=time_train,
            weight=train_group_weight,
            max_bin=config.xgb_max_bin,
            nthread=config.num_threads,
        )
        dstop = xgb.QuantileDMatrix(
            x_stop,
            label=y_stop,
            qid=time_stop,
            weight=stop_group_weight,
            max_bin=config.xgb_max_bin,
            nthread=config.num_threads,
            ref=dtrain,
        )
        del x_train, x_stop

        def exact_metric(prediction: np.ndarray, _dataset: Any) -> tuple[str, float]:
            return "tsauc", float(tsauc(y_stop, prediction, time_stop))

        parameters = {
            "objective": "rank:pairwise",
            "tree_method": "hist",
            "device": "cpu",
            "disable_default_eval_metric": 1,
            "eta": config.xgb_learning_rate,
            "max_depth": config.xgb_max_depth,
            "min_child_weight": config.xgb_min_child_weight,
            "subsample": config.xgb_subsample,
            "colsample_bytree": config.xgb_colsample,
            "alpha": config.xgb_reg_alpha,
            "lambda": config.xgb_reg_lambda,
            "max_bin": config.xgb_max_bin,
            "lambdarank_pair_method": config.pair_method,
            "lambdarank_num_pair_per_sample": config.pairs_per_sample,
            "lambdarank_normalization": True,
            "seed": stage_seed,
            "nthread": config.num_threads,
            "verbosity": 0,
        }
        booster = xgb.train(
            parameters,
            dtrain,
            num_boost_round=config.boosting_rounds,
            evals=[(dstop, "stop")],
            custom_metric=exact_metric,
            maximize=True,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=False,
        )
        best_iteration = int(booster.best_iteration)
        stop_raw = booster.predict(
            dstop, iteration_range=(0, best_iteration + 1)
        )
        center, scale = _calibration(stop_raw)
        prediction = _predict_xgb_rows(
            context,
            booster,
            score_rows,
            columns,
            best_iteration,
            center,
            scale,
        )
        booster.save_model(model_path)
        actual_train_rows = int(query_train.size)
        actual_stop_rows = int(query_stop.size)
        query_groups = {
            "train": int(train_group_weight.size),
            "stop": int(stop_group_weight.size),
        }
        del dtrain, dstop, booster, stop_raw
    elif spec.kind == "lightgbm_goss":
        x_train = _sanitize(
            matrix_rows(context.source.data.joint_matrix, train_rows, columns)
        )
        x_stop = _sanitize(
            matrix_rows(context.source.data.joint_matrix, stop_rows, columns)
        )
        y_train = metadata["target"][train_rows]
        y_stop = metadata["target"][stop_rows]
        time_stop = metadata["time_online"][stop_rows]
        train_weight = pair_weights_from_reference(
            y_train,
            metadata["time_online"][train_rows],
            metadata["target"][train_role],
            metadata["time_online"][train_role],
        )
        stop_weight = pair_weights_from_reference(
            y_stop,
            time_stop,
            metadata["target"][stop_role],
            metadata["time_online"][stop_role],
        )
        train_set = lgb.Dataset(x_train, label=y_train, weight=train_weight)
        stop_set = lgb.Dataset(
            x_stop, label=y_stop, weight=stop_weight, reference=train_set
        )

        def evaluator(values: np.ndarray, dataset: lgb.Dataset):
            return "tsauc", tsauc(dataset.get_label(), values, time_stop), True

        parameters = {
            "objective": "binary",
            "metric": "None",
            "boosting_type": "goss",
            "learning_rate": config.goss_learning_rate,
            "num_leaves": config.goss_num_leaves,
            "min_data_in_leaf": config.goss_min_data_in_leaf,
            "feature_fraction": config.goss_feature_fraction,
            "lambda_l1": config.goss_lambda_l1,
            "lambda_l2": config.goss_lambda_l2,
            "top_rate": config.goss_top_rate,
            "other_rate": config.goss_other_rate,
            "verbosity": -1,
            "num_threads": config.num_threads,
            "seed": stage_seed,
            "feature_fraction_seed": stage_seed + 1,
            "data_random_seed": stage_seed + 2,
            "deterministic": True,
            "force_col_wise": True,
        }
        booster = lgb.train(
            parameters,
            train_set,
            num_boost_round=config.boosting_rounds,
            valid_sets=[stop_set],
            feval=evaluator,
            callbacks=[
                lgb.early_stopping(
                    config.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(booster.best_iteration or config.boosting_rounds)
        stop_probability = booster.predict(x_stop, num_iteration=best_iteration)
        center, scale = _calibration(_safe_logit(stop_probability))
        prediction = _predict_lgb_rows(
            context,
            booster,
            score_rows,
            columns,
            best_iteration,
            center,
            scale,
        )
        booster.save_model(model_path, num_iteration=best_iteration)
        actual_train_rows = int(train_rows.size)
        actual_stop_rows = int(stop_rows.size)
        query_groups = None
        del (
            x_train,
            x_stop,
            y_train,
            y_stop,
            train_weight,
            stop_weight,
            train_set,
            stop_set,
            booster,
            stop_probability,
        )
    else:
        raise ValueError(f"unsupported model kind: {spec.kind}")

    target = metadata["target"][score_rows]
    score = float(tsauc(target, prediction, metadata["time_online"][score_rows]))
    record = {
        **contract,
        "contract_hash": contract_hash,
        "tsauc": score,
        "ordinary_roc_diagnostic": float(roc_auc_score(target, prediction)),
        "best_iteration": best_iteration,
        "sampled_complete_series_train_rows": int(train_rows.size),
        "sampled_complete_series_stop_rows": int(stop_rows.size),
        "effective_train_rows": actual_train_rows,
        "effective_stop_rows": actual_stop_rows,
        "query_groups": query_groups,
        "score_rows": int(score_rows.size),
        "calibration_center": center,
        "calibration_scale": scale,
        "elapsed_seconds": float(time.perf_counter() - started),
        "prediction_path": str(prediction_path.relative_to(config.project_root)),
        "model_path": str(model_path.relative_to(config.project_root)),
    }
    np.save(prediction_path, prediction.astype(np.float32))
    np.savez_compressed(
        columns_path,
        columns=columns,
        names=np.asarray(names, dtype=np.str_),
    )
    _atomic_json(result_path, record)
    print(
        f"{spec.name}/repeat{repeat}/fold{fold}: TSAUC={score:.6f}, "
        f"best_iteration={best_iteration}, {record['elapsed_seconds']/60:.2f} min",
        flush=True,
    )
    del prediction, target
    gc.collect()
    return record


def run_screen(context: Model7Context) -> dict[str, Any]:
    """Run the declared two-fold timing screen without dropping any model."""

    records: dict[str, Any] = {}
    for spec in model_specs(context.config):
        rows = [
            fit_fold(context, spec, context.config.screen_repeat, fold)
            for fold in context.config.screen_folds
        ]
        scores = [row["tsauc"] for row in rows]
        records[spec.name] = {
            "model": spec.__dict__,
            "tsroc": _mean_se(scores),
            "robust_score": robust_score(
                scores, context.config.uncertainty_penalty
            ),
            "elapsed_minutes": float(
                sum(row["elapsed_seconds"] for row in rows) / 60.0
            ),
        }
    report = {
        "stage": "informational_screen",
        "selection_effect": "none; every predeclared model continues to repeated CV",
        "repeat": context.config.screen_repeat,
        "folds": list(context.config.screen_folds),
        "models": records,
        "outer_holdout_read": False,
    }
    _atomic_json(context.config.artifact_path / "screen/summary.json", report)
    return report


def _assemble_repeat_oof(
    context: Model7Context, spec: ModelSpec, repeat: int
) -> np.ndarray:
    path = context.config.artifact_path / "oof" / spec.name / f"repeat_{repeat}.f32.npy"
    if path.exists():
        prediction = np.load(path, allow_pickle=False)
        if prediction.shape != (context.development_rows.size,):
            raise ValueError(f"wrong OOF shape: {path}")
        return prediction
    prediction = np.full(context.development_rows.size, np.nan, dtype=np.float32)
    fold_value = context.repeat_row_fold[repeat]
    for fold in range(context.config.folds_per_repeat):
        record = fit_fold(context, spec, repeat, fold)
        saved = np.load(
            context.config.project_root / record["prediction_path"], allow_pickle=False
        )
        mask = fold_value == fold
        if saved.size != int(mask.sum()):
            raise ValueError("fold prediction length differs from held rows")
        prediction[mask] = saved
    if not np.all(np.isfinite(prediction)):
        raise ValueError("OOF prediction is incomplete or non-finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prediction)
    return prediction


def _model_cv_report(
    context: Model7Context, spec: ModelSpec, predictions: np.ndarray
) -> dict[str, Any]:
    metadata = context.source.data.metadata
    target = metadata["target"][context.development_rows]
    time_online = metadata["time_online"][context.development_rows]
    repeat_fold_scores: list[list[float]] = []
    for repeat in range(len(context.config.repeat_seeds)):
        scores = []
        for fold in range(context.config.folds_per_repeat):
            mask = context.repeat_row_fold[repeat] == fold
            scores.append(
                float(
                    tsauc(
                        target[mask], predictions[repeat, mask], time_online[mask]
                    )
                )
            )
        repeat_fold_scores.append(scores)
    repeat_means = [float(np.mean(scores)) for scores in repeat_fold_scores]
    averaged = predictions.mean(axis=0)
    return {
        "model": spec.__dict__,
        "repeat_fold_scores": repeat_fold_scores,
        "repeat_means": _mean_se(repeat_means),
        "all_fold_scores_diagnostic": _mean_se(
            [score for scores in repeat_fold_scores for score in scores]
        ),
        "robust_repeat_score": robust_score(
            repeat_means, context.config.uncertainty_penalty
        ),
        "pooled_average_repeat_oof": float(tsauc(target, averaged, time_online)),
        "ordinary_roc_diagnostic": float(roc_auc_score(target, averaged)),
        "oof_paths": [
            str(
                (
                    context.config.artifact_path
                    / "oof"
                    / spec.name
                    / f"repeat_{repeat}.f32.npy"
                ).relative_to(context.config.project_root)
            )
            for repeat in range(len(context.config.repeat_seeds))
        ],
    }


def run_repeated_cv(context: Model7Context) -> dict[str, Any]:
    """Fit every predeclared model on every repeated held fold."""

    reports: dict[str, Any] = {}
    for spec in model_specs(context.config):
        repeat_predictions = []
        for repeat in range(len(context.config.repeat_seeds)):
            for fold in range(context.config.folds_per_repeat):
                fit_fold(context, spec, repeat, fold)
            repeat_predictions.append(_assemble_repeat_oof(context, spec, repeat))
        matrix = np.stack(repeat_predictions, axis=0)
        reports[spec.name] = _model_cv_report(context, spec, matrix)
        print(
            f"{spec.name}: repeat mean "
            f"{reports[spec.name]['repeat_means']['mean']:.6f} +/- "
            f"{reports[spec.name]['repeat_means']['standard_error']:.6f}",
            flush=True,
        )
    report = {
        "stage": "three_repeat_five_fold_cv",
        "selection_metric": "TSAUC",
        "uncertainty_unit": "standard error across three repeat means",
        "models": reports,
        "outer_holdout_read": False,
    }
    _atomic_json(context.config.artifact_path / "repeated_cv/summary.json", report)
    return report


def _blend(first: np.ndarray, second: np.ndarray, second_weight: float) -> np.ndarray:
    margin = (1.0 - second_weight) * _safe_logit(first) + second_weight * _safe_logit(
        second
    )
    return _logistic(margin).astype(np.float32)


def _fold_scores(
    target: np.ndarray,
    prediction: np.ndarray,
    time_online: np.ndarray,
    fold: np.ndarray,
    included_folds: Sequence[int],
) -> list[float]:
    return [
        float(
            tsauc(
                target[fold == value],
                prediction[fold == value],
                time_online[fold == value],
            )
        )
        for value in included_folds
    ]


def _nested_low_capacity_blend(
    context: Model7Context, component_predictions: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose at most two components and one weight without each held fold."""

    config = context.config
    metadata = context.source.data.metadata
    target = metadata["target"][context.development_rows]
    time_online = metadata["time_online"][context.development_rows]
    output = np.empty(
        (len(config.repeat_seeds), context.development_rows.size), dtype=np.float32
    )
    held_recipes = []
    for repeat in range(len(config.repeat_seeds)):
        fold_value = context.repeat_row_fold[repeat]
        for held in range(config.folds_per_repeat):
            tune_folds = [
                fold for fold in range(config.folds_per_repeat) if fold != held
            ]
            individual: dict[str, float] = {}
            for name, values in component_predictions.items():
                scores = _fold_scores(
                    target,
                    values[repeat],
                    time_online,
                    fold_value,
                    tune_folds,
                )
                individual[name] = robust_score(
                    scores, config.uncertainty_penalty
                )
            ordered = sorted(individual, key=lambda name: (-individual[name], name))
            first = ordered[0]
            second = ordered[1]
            best: tuple[float, float, list[float]] | None = None
            for weight in config.blend_weights:
                prediction = _blend(
                    component_predictions[first][repeat],
                    component_predictions[second][repeat],
                    weight,
                )
                scores = _fold_scores(
                    target, prediction, time_online, fold_value, tune_folds
                )
                criterion = robust_score(scores, config.uncertainty_penalty)
                row = (criterion, -float(weight), scores)
                if best is None or row[:2] > best[:2]:
                    best = row
            if best is None:
                raise AssertionError("blend grid is empty")
            selected_weight = -best[1]
            prediction = _blend(
                component_predictions[first][repeat],
                component_predictions[second][repeat],
                selected_weight,
            )
            held_mask = fold_value == held
            output[repeat, held_mask] = prediction[held_mask]
            held_score = float(
                tsauc(
                    target[held_mask],
                    output[repeat, held_mask],
                    time_online[held_mask],
                )
            )
            held_recipes.append(
                {
                    "repeat": repeat,
                    "held_fold": held,
                    "first": first,
                    "second": second,
                    "second_weight": selected_weight,
                    "tuning_fold_scores": best[2],
                    "tuning_robust_score": best[0],
                    "held_tsroc": held_score,
                }
            )
    return output, {"method": "nested two-component robust logit blend", "held_recipes": held_recipes}


def _weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    denominator = max(float(weights.sum()), 1e-12)
    mean = float(np.sum(values * weights) / denominator)
    variance = float(np.sum((values - mean) ** 2 * weights) / denominator)
    return mean, max(math.sqrt(max(variance, 0.0)), 0.20)


def _fit_absorbing_calibration(
    context: Model7Context,
    probability: np.ndarray,
    tune_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit time-binned emissions and a one-way break hazard on tuning series."""

    metadata = context.source.data.metadata
    dev = context.development_rows
    y = metadata["target"][dev][tune_mask]
    time_online = metadata["time_online"][dev][tune_mask]
    margin = np.clip(_safe_logit(probability[tune_mask]), -8.0, 8.0)
    weights = pair_weights_from_reference(y, time_online, y, time_online).astype(
        np.float64
    )
    global_moments = [
        _weighted_moments(margin[y == label], weights[y == label])
        for label in (0, 1)
    ]
    edges = np.asarray(context.config.hmm_time_edges, dtype=np.int64)
    bins = np.clip(np.searchsorted(edges, time_online, side="right") - 1, 0, edges.size - 2)
    mean = np.zeros((2, edges.size - 1), dtype=np.float64)
    scale = np.zeros_like(mean)
    for label in (0, 1):
        for time_bin in range(edges.size - 1):
            selected = (y == label) & (bins == time_bin) & (weights > 0)
            if int(selected.sum()) < 100:
                mean[label, time_bin], scale[label, time_bin] = global_moments[label]
                continue
            local_mean, local_scale = _weighted_moments(
                margin[selected], weights[selected]
            )
            shrink = float(selected.sum() / (selected.sum() + 1_000.0))
            mean[label, time_bin] = (
                shrink * local_mean + (1.0 - shrink) * global_moments[label][0]
            )
            scale[label, time_bin] = max(
                shrink * local_scale + (1.0 - shrink) * global_moments[label][1],
                0.20,
            )

    tune_series = np.unique(metadata["series_position"][dev][tune_mask])
    lengths = metadata["online_lengths"][tune_series]
    tau = metadata["tau_index"][tune_series]
    maximum = int(metadata["time_online"][dev].max(initial=0))
    hazard = np.empty(maximum + 1, dtype=np.float64)
    for online_time in range(maximum + 1):
        at_risk = np.sum((lengths > online_time) & ((tau < 0) | (tau >= online_time)))
        transitions = np.sum(tau == online_time)
        hazard[online_time] = (transitions + 0.5) / (at_risk + 50.0)
    return {"edges": edges, "mean": mean, "scale": scale, "hazard": hazard}


def apply_absorbing_filter_grid(
    probability: np.ndarray,
    time_online: np.ndarray,
    series_position: np.ndarray,
    calibration: dict[str, np.ndarray],
    evidence_scales: Sequence[float],
    hazard_scales: Sequence[float],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Apply all causal absorbing-state settings in one chronological pass.

    Series are conditionally independent, so rows at the same ``time_online``
    can be updated together.  Looping over time slices is mathematically the
    same as looping over series and then time, but avoids millions of Python
    iterations during the nested filter search.
    """

    probability = np.asarray(probability, dtype=np.float64)
    time_online = np.asarray(time_online, dtype=np.int64)
    series_position = np.asarray(series_position, dtype=np.int64)
    if not (probability.shape == time_online.shape == series_position.shape):
        raise ValueError("absorbing-filter vectors have different shapes")
    settings = [
        (float(evidence), float(hazard))
        for evidence in evidence_scales
        for hazard in hazard_scales
    ]
    evidence = np.asarray([row[0] for row in settings], dtype=np.float64)
    hazard_multiplier = np.asarray([row[1] for row in settings], dtype=np.float64)
    output = np.empty((len(settings), probability.size), dtype=np.float32)
    margin = np.clip(_safe_logit(probability), -8.0, 8.0)
    edges = calibration["edges"]
    means = calibration["mean"]
    scales = calibration["scale"]
    hazard = calibration["hazard"]
    _, dense_series = np.unique(series_position, return_inverse=True)
    state = np.full(
        (len(settings), int(dense_series.max(initial=-1)) + 1),
        1e-5,
        dtype=np.float64,
    )
    order = np.argsort(time_online, kind="stable")
    sorted_time = time_online[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_time)) + 1]
    stops = np.r_[starts[1:], order.size]
    for start, stop in zip(starts, stops, strict=True):
        positions = order[start:stop]
        online_time = int(sorted_time[start])
        active_series = dense_series[positions]
        if np.unique(active_series).size != active_series.size:
            raise ValueError(
                "absorbing filter requires at most one row per series/time pair"
            )
        time_bin = int(
            np.clip(
                np.searchsorted(edges, online_time, side="right") - 1,
                0,
                edges.size - 2,
            )
        )
        values = margin[positions]
        log_likelihood_ratio = (
            -0.5 * ((values - means[1, time_bin]) / scales[1, time_bin]) ** 2
            - math.log(scales[1, time_bin])
            + 0.5 * ((values - means[0, time_bin]) / scales[0, time_bin]) ** 2
            + math.log(scales[0, time_bin])
        )
        transition = np.clip(
            hazard[min(online_time, hazard.size - 1)] * hazard_multiplier,
            1e-7,
            0.25,
        )[:, None]
        active_state = state[:, active_series]
        prior = np.clip(
            active_state + (1.0 - active_state) * transition,
            1e-7,
            1 - 1e-7,
        )
        posterior_logit = (
            np.log(prior)
            - np.log1p(-prior)
            + evidence[:, None]
            * np.clip(log_likelihood_ratio[None, :], -12.0, 12.0)
        )
        posterior = _logistic(posterior_logit)
        state[:, active_series] = posterior
        output[:, positions] = posterior.astype(np.float32)
    return output, settings


def _nested_absorbing_filter(
    context: Model7Context, raw_prediction: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    config = context.config
    metadata = context.source.data.metadata
    dev = context.development_rows
    target = metadata["target"][dev]
    time_online = metadata["time_online"][dev]
    series_position = metadata["series_position"][dev]
    output = np.empty_like(raw_prediction, dtype=np.float32)
    held_recipes = []
    for repeat in range(len(config.repeat_seeds)):
        fold_value = context.repeat_row_fold[repeat]
        for held in range(config.folds_per_repeat):
            held_mask = fold_value == held
            tune_mask = ~held_mask
            calibration = _fit_absorbing_calibration(
                context, raw_prediction[repeat], tune_mask
            )
            grid, settings = apply_absorbing_filter_grid(
                raw_prediction[repeat],
                time_online,
                series_position,
                calibration,
                config.hmm_evidence_scales,
                config.hmm_hazard_scales,
            )
            tune_folds = [
                fold for fold in range(config.folds_per_repeat) if fold != held
            ]
            candidates: list[tuple[str, np.ndarray, float, list[float]]] = []
            raw_scores = _fold_scores(
                target,
                raw_prediction[repeat],
                time_online,
                fold_value,
                tune_folds,
            )
            candidates.append(
                (
                    "raw",
                    raw_prediction[repeat],
                    robust_score(raw_scores, config.uncertainty_penalty),
                    raw_scores,
                )
            )
            for index, (evidence, hazard) in enumerate(settings):
                scores = _fold_scores(
                    target, grid[index], time_online, fold_value, tune_folds
                )
                candidates.append(
                    (
                        f"hmm_e{evidence:g}_h{hazard:g}",
                        grid[index],
                        robust_score(scores, config.uncertainty_penalty),
                        scores,
                    )
                )
            selected = max(candidates, key=lambda row: (row[2], row[0] == "raw", row[0]))
            output[repeat, held_mask] = selected[1][held_mask]
            held_score = float(
                tsauc(
                    target[held_mask],
                    output[repeat, held_mask],
                    time_online[held_mask],
                )
            )
            held_recipes.append(
                {
                    "repeat": repeat,
                    "held_fold": held,
                    "selected": selected[0],
                    "tuning_fold_scores": selected[3],
                    "tuning_robust_score": selected[2],
                    "held_tsroc": held_score,
                }
            )
            del grid
            gc.collect()
    return output, {
        "method": "nested causal Gaussian-emission absorbing-state filter",
        "raw_is_an_allowed_endpoint": True,
        "held_recipes": held_recipes,
    }


def _architecture_report(
    context: Model7Context, prediction: np.ndarray
) -> dict[str, Any]:
    metadata = context.source.data.metadata
    dev = context.development_rows
    target = metadata["target"][dev]
    time_online = metadata["time_online"][dev]
    repeat_fold_scores = []
    for repeat in range(len(context.config.repeat_seeds)):
        fold_scores = []
        for fold in range(context.config.folds_per_repeat):
            mask = context.repeat_row_fold[repeat] == fold
            fold_scores.append(
                float(
                    tsauc(
                        target[mask], prediction[repeat, mask], time_online[mask]
                    )
                )
            )
        repeat_fold_scores.append(fold_scores)
    repeat_means = [float(np.mean(scores)) for scores in repeat_fold_scores]
    averaged = prediction.mean(axis=0)
    return {
        "repeat_fold_scores": repeat_fold_scores,
        "repeat_means": _mean_se(repeat_means),
        "robust_repeat_score": robust_score(
            repeat_means, context.config.uncertainty_penalty
        ),
        "pooled_average_repeat_oof": float(tsauc(target, averaged, time_online)),
        "ordinary_roc_diagnostic": float(roc_auc_score(target, averaged)),
    }


def run_ensemble(context: Model7Context) -> dict[str, Any]:
    """Build the nested two-model blend and optional absorbing-state filter."""

    repeated_path = context.config.artifact_path / "repeated_cv/summary.json"
    if not repeated_path.exists():
        run_repeated_cv(context)
    components = {
        spec.name: np.stack(
            [
                _assemble_repeat_oof(context, spec, repeat)
                for repeat in range(len(context.config.repeat_seeds))
            ],
            axis=0,
        )
        for spec in model_specs(context.config)
    }
    raw, blend_report = _nested_low_capacity_blend(context, components)
    filtered, filter_report = _nested_absorbing_filter(context, raw)
    raw_report = _architecture_report(context, raw)
    filtered_report = _architecture_report(context, filtered)
    raw_path = context.config.artifact_path / "ensemble/raw_nested.f32.npy"
    filtered_path = context.config.artifact_path / "ensemble/filtered_nested.f32.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, raw)
    np.save(filtered_path, filtered)

    # The filter's held-fold selector already includes raw. Select the final
    # reporting architecture by repeated-CV robust score, with raw winning ties.
    if filtered_report["robust_repeat_score"] > raw_report["robust_repeat_score"] + 1e-12:
        final_name = "nested_absorbing_filter"
        final_prediction = filtered
        final_report = filtered_report
    else:
        final_name = "raw_nested_blend"
        final_prediction = raw
        final_report = raw_report
    metadata = context.source.data.metadata
    dev = context.development_rows
    averaged = final_prediction.mean(axis=0)
    bootstrap = base.cluster_bootstrap_tsauc(
        metadata["target"][dev],
        averaged,
        metadata["time_online"][dev],
        metadata["series_position"][dev],
        replicates=context.config.bootstrap_replicates,
        seed=context.config.bootstrap_seed,
    )
    report = {
        "stage": "nested_robust_ensemble",
        "selection_metric": "mean repeat TSAUC - uncertainty_penalty * repeat SE",
        "outer_holdout_read": False,
        "blend": {**blend_report, "score": raw_report},
        "absorbing_filter": {**filter_report, "score": filtered_report},
        "selected_architecture": final_name,
        "selected_score": final_report,
        "complete_series_bootstrap_of_average_repeat_oof": bootstrap,
        "target_tsroc": context.config.target_tsroc,
        "target_reached_on_repeated_development_cv": bool(
            final_report["repeat_means"]["mean"] >= context.config.target_tsroc
        ),
        "estimate_warning": (
            "All values use Model 6 development series. Model 6's outer holdout "
            "has been revealed and is intentionally never scored by Model 7."
        ),
    }
    _atomic_json(context.config.artifact_path / "ensemble/summary.json", report)
    return report


def run_all(context: Model7Context) -> dict[str, Any]:
    """Run or resume every train-only Model 7 stage."""

    run_screen(context)
    run_repeated_cv(context)
    return run_ensemble(context)
