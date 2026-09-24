"""Leakage-controlled Model 6 search and ensemble pipeline.

The pipeline deliberately separates four kinds of data use:

* Model 5's five development OOF folds select detector banks by TSROC;
* a held development fold scores each feature/model candidate while a
  different fold supplies early stopping;
* only promoted candidates receive broad hyperparameter tuning and full CV;
* the outer holdout remains unavailable until a hashed deployment contract is
  frozen.

Every expensive fit has a JSON checkpoint.  Re-running the CLI resumes at the
first missing compatible checkpoint rather than overwriting completed work.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from models.model4 import model4_pipeline as base
from models.model5.config import make_model5_config
from models.model5.model5_pipeline import load_model5_data, prepare_split
from models.model6.config import MODEL6_VERSION, Model6Config
from postprocess import causal_decayed_max, causal_probability_decayed_max
from train_cv import matrix_rows, pair_weights_from_reference
from tsauc import fold_standard_error, tsauc


FAMILIES = ("core", "candidate")
STACK_MODEL_TYPES = ("linear_l2", "linear_l1", "lightgbm_gbdt", "extra_trees")


@dataclass
class Model6Context:
    """Loaded train-derived matrices and immutable Model 5 source contracts."""

    config: Model6Config
    source_config: Any
    data: Any
    split: Any
    source_selection: dict[str, Any]
    family_columns: dict[str, np.ndarray]
    family_names: dict[str, list[str]]
    family_banks: dict[str, tuple[str, ...]]
    actual_device: str


@dataclass
class FittedModel:
    """One backend-neutral fitted learner."""

    model_type: str
    model: Any
    scaler: Any = None
    best_iteration: int = 0


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
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != ready:
            raise ValueError(
                f"{path} belongs to a different Model 6 protocol; change the "
                "configuration so it receives a different run directory"
            )
        return
    _atomic_json(path, ready)


def _mean_se(scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite and non-empty")
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


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(bounded) - np.log1p(-bounded)


def _logistic(margin: np.ndarray) -> np.ndarray:
    value = np.asarray(margin, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _sanitize(matrix: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(matrix, dtype=np.float32)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return output


def _rank_descending(values: np.ndarray) -> np.ndarray:
    score = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=-np.inf,
        neginf=-np.inf,
        posinf=np.finfo(np.float64).max,
    )
    return np.lexsort((np.arange(score.size), -score)).astype(np.int32)


def _rank_positions(order: np.ndarray) -> np.ndarray:
    result = np.empty(order.size, dtype=np.float64)
    result[np.asarray(order, dtype=np.int32)] = np.arange(order.size)
    return result


def _cap_rows(rows: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Uniform exact-size selection independent of labels and feature values."""

    rows = np.asarray(rows, dtype=np.int64)
    if rows.size <= maximum:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=maximum, replace=False))


def _working_cuda_available() -> tuple[bool, str]:
    """Conservatively detect an NVIDIA driver, not merely CUDA Python wheels."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, "nvidia-smi is not installed; no NVIDIA runtime is visible"
    try:
        probe = subprocess.run(
            [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"nvidia-smi probe failed: {error}"
    if probe.returncode or not probe.stdout.strip():
        detail = (probe.stderr or probe.stdout).strip().replace("\n", " ")
        return False, f"NVIDIA driver probe failed: {detail or probe.returncode}"
    return True, probe.stdout.strip()


def load_context(config: Model6Config) -> Model6Context:
    """Load Model 5's selected banks and write Model 6's source manifest."""

    config.validate()
    source = make_model5_config(config.project_root)
    if not math.isclose(
        source.detector_roc_threshold, config.source_detector_threshold
    ) or not math.isclose(
        source.detector_standard_error_multiplier,
        config.source_detector_se_multiplier,
    ):
        raise ValueError(
            "Model 5's editable TSROC detector rule differs from Model 6's "
            "declared source rule"
        )
    data = load_model5_data(source)
    split = prepare_split(data, source)
    selection_path = source.artifact_path / "selected_detectors.json"
    if not selection_path.exists():
        raise FileNotFoundError("finish Model 5 detector selection before Model 6")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    family_columns: dict[str, np.ndarray] = {}
    family_names: dict[str, list[str]] = {}
    family_banks: dict[str, tuple[str, ...]] = {}
    for family in FAMILIES:
        columns, names, banks = base.load_selection_arrays(source, family)
        family_columns[family] = columns
        family_names[family] = names
        family_banks[family] = banks

    has_cuda, device_detail = _working_cuda_available()
    if config.requested_device == "cuda" and not has_cuda:
        raise RuntimeError(f"CUDA was required but is unusable: {device_detail}")
    actual_device = "cuda" if has_cuda and config.requested_device != "cpu" else "cpu"
    manifest = {
        "experiment_version": MODEL6_VERSION,
        "protocol": config.protocol(),
        "protocol_hash": _hash_json(config.protocol()),
        "source_model5_artifact": str(source.artifact_path.relative_to(config.project_root)),
        "source_selection_hash": selection["selection_hash"],
        "source_selection_rule": selection["selection_rule"],
        "selected_core_banks": selection["selected_core_banks"],
        "selected_candidate_banks": selection["selected_candidate_banks"],
        "family_feature_counts": {
            family: int(family_columns[family].size) for family in FAMILIES
        },
        "split": {
            "outer_holdout_fold": source.holdout_fold,
            "development_folds": source.development_folds,
            "complete_series_only": True,
        },
        "device": {
            "requested": config.requested_device,
            "actual": actual_device,
            "probe": device_detail,
            "cpu_threads": config.num_threads,
        },
        "holdout_labels_read": False,
    }
    _write_or_validate(config.artifact_path / "experiment_manifest.json", manifest)
    return Model6Context(
        config=config,
        source_config=source,
        data=data,
        split=split,
        source_selection=selection,
        family_columns=family_columns,
        family_names=family_names,
        family_banks=family_banks,
        actual_device=actual_device,
    )


def context_summary(context: Model6Context) -> dict[str, Any]:
    """Small notebook/CLI summary that cannot expose holdout outcomes."""

    return {
        "run": context.config.artifact_directory,
        "device": context.actual_device,
        "development_series": int(context.split.development_series_mask.sum()),
        "sealed_holdout_series": int(context.split.holdout_series_mask.sum()),
        "development_rows": int(context.split.development_row_mask.sum()),
        "core_banks": list(context.family_banks["core"]),
        "candidate_banks": list(context.family_banks["candidate"]),
        "core_features": int(context.family_columns["core"].size),
        "candidate_features": int(context.family_columns["candidate"].size),
    }


def _fisher_scores(
    context: Model6Context,
    family: str,
    train_mask: np.ndarray,
    fold: int,
) -> np.ndarray:
    """Weighted standardized class separation on training-role rows only."""

    config = context.config
    eligible = np.flatnonzero(train_mask & context.split.sampled_row_mask)
    rows = _cap_rows(eligible, config.ranking_sample_rows, config.seed + 101 * fold)
    y = context.data.metadata["target"][rows]
    weights = pair_weights_from_reference(
        y,
        context.data.metadata["time_online"][rows],
        context.data.metadata["target"][train_mask],
        context.data.metadata["time_online"][train_mask],
    ).astype(np.float64)
    result = np.zeros(context.family_columns[family].size, dtype=np.float64)
    for left in range(0, result.size, 256):
        right = min(left + 256, result.size)
        x = _sanitize(
            matrix_rows(
                context.data.joint_matrix,
                rows,
                context.family_columns[family][left:right],
            )
        ).astype(np.float64)
        class_values = []
        for label in (0, 1):
            selected = y == label
            w = weights[selected]
            denominator = max(float(w.sum()), 1e-12)
            mean = np.sum(x[selected] * w[:, None], axis=0) / denominator
            variance = (
                np.sum((x[selected] - mean) ** 2 * w[:, None], axis=0)
                / denominator
            )
            class_values.append((mean, variance))
        result[left:right] = np.abs(class_values[1][0] - class_values[0][0]) / np.sqrt(
            class_values[1][1] + class_values[0][1] + 1e-8
        )
        del x
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _correlation_pruned_order(
    context: Model6Context,
    family: str,
    train_mask: np.ndarray,
    gain_order: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Move highly correlated gain leaders behind non-redundant leaders."""

    config = context.config
    pool_size = min(config.correlation_pool_size, gain_order.size)
    pool = gain_order[:pool_size]
    eligible = np.flatnonzero(train_mask)
    rows = _cap_rows(
        eligible,
        config.correlation_sample_rows,
        config.seed + 20_000 + fold,
    )
    x = _sanitize(
        matrix_rows(
            context.data.joint_matrix,
            rows,
            context.family_columns[family][pool],
        )
    ).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(x * x, axis=0))
    norms[~np.isfinite(norms) | (norms < 1e-12)] = np.inf
    x /= norms
    correlation = np.abs(x.T @ x)
    np.fill_diagonal(correlation, 0.0)
    kept_positions: list[int] = []
    for candidate in range(pool_size):
        if not kept_positions or float(
            np.max(correlation[candidate, kept_positions])
        ) < config.correlation_abs_threshold:
            kept_positions.append(candidate)
    kept = pool[np.asarray(kept_positions, dtype=np.int32)]
    used = np.zeros(gain_order.size, dtype=bool)
    used[kept] = True
    order = np.concatenate((kept, gain_order[~used[gain_order]])).astype(np.int32)
    del x, correlation
    gc.collect()
    return order, {
        "correlation_rows": int(rows.size),
        "correlation_pool": int(pool_size),
        "nonredundant_pool_features": int(kept.size),
    }


def build_feature_rankings(context: Model6Context) -> None:
    """Create five score-fold-excluded rankings for both feature families.

    Gain and split importances reuse Model 5's compatible all-bank baseline
    models.  Those models have the identical split and exclude the current
    scoring fold; current TSROC-selected columns are mapped by exact feature
    name.  Fisher and correlation calculations use only that fold's fitting
    role.
    """

    config = context.config
    baseline_root = context.source_config.artifact_path / "combined_cv"
    for family in FAMILIES:
        names = context.family_names[family]
        for fold in range(5):
            output = config.artifact_path / "rankings" / family / f"fold_{fold}.npz"
            report_path = output.with_suffix(".json")
            source_model = baseline_root / family / f"fold_{fold}.txt"
            source_manifest = baseline_root / family / "manifest.json"
            if not source_model.exists() or not source_manifest.exists():
                raise FileNotFoundError(
                    "Model 5 family baseline models are required for leakage-safe "
                    f"importance reuse: {source_model}"
                )
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            source_names = manifest["feature_names"]
            source_position = {name: index for index, name in enumerate(source_names)}
            if len(source_position) != len(source_names) or any(
                name not in source_position for name in names
            ):
                raise ValueError("Model 5 baseline cannot map current feature names exactly")
            contract = {
                "stage": "feature_ranking",
                "family": family,
                "fold": fold,
                "score_fold_excluded": True,
                "source_model": str(source_model.relative_to(config.project_root)),
                "source_feature_name_mapping": "exact name; selected TSROC subset only",
                "source_selection_hash": context.source_selection["selection_hash"],
                "methods": list(config.ranking_methods),
                "ranking_sample_rows": config.ranking_sample_rows,
                "correlation_sample_rows": config.correlation_sample_rows,
                "correlation_pool_size": config.correlation_pool_size,
                "correlation_abs_threshold": config.correlation_abs_threshold,
            }
            contract_hash = _hash_json(contract)
            if output.exists() and report_path.exists():
                saved_report = json.loads(report_path.read_text(encoding="utf-8"))
                if saved_report.get("contract_hash") != contract_hash:
                    raise ValueError(f"stale ranking checkpoint: {output}")
                print(f"ranking {family} fold {fold}: already complete", flush=True)
                continue

            train_mask, _, _, stop_fold = base.development_fold_roles(
                context.split, fold, 5
            )
            booster = lgb.Booster(model_file=str(source_model))
            positions = np.asarray([source_position[name] for name in names], dtype=np.int32)
            gain = booster.feature_importance("gain")[positions]
            splits = booster.feature_importance("split")[positions]
            fisher = _fisher_scores(context, family, train_mask, fold)
            gain_order = _rank_descending(gain)
            split_order = _rank_descending(splits)
            fisher_order = _rank_descending(fisher)
            consensus_value = (
                _rank_positions(gain_order)
                + _rank_positions(split_order)
                + _rank_positions(fisher_order)
            )
            consensus_order = np.lexsort(
                (np.arange(consensus_value.size), consensus_value)
            ).astype(np.int32)
            gain_corr, corr_report = _correlation_pruned_order(
                context, family, train_mask, gain_order, fold
            )
            rankings = {
                "gain": gain_order,
                "split": split_order,
                "consensus": consensus_order,
                "gain_corr": gain_corr,
                "fisher": fisher_order,
            }
            for method, order in rankings.items():
                if not np.array_equal(np.sort(order), np.arange(len(names))):
                    raise AssertionError(f"{family}/{fold}/{method} is not a permutation")
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **rankings)
            _atomic_json(
                report_path,
                {
                    **contract,
                    **corr_report,
                    "early_stop_fold_of_source_model": stop_fold,
                    "feature_count": len(names),
                    "contract_hash": contract_hash,
                },
            )
            print(f"ranking {family} fold {fold}: complete", flush=True)
            del booster, gain, splits, fisher
            gc.collect()


def feature_contracts(config: Model6Config) -> list[dict[str, Any]]:
    """Return every requested ranking/count plus one non-duplicated all set."""

    contracts = [
        {"method": method, "count": int(count), "key": f"{method}_{count}"}
        for method in config.ranking_methods
        for count in config.feature_counts
    ]
    if config.include_all_features:
        contracts.append({"method": "all", "count": "all", "key": "all"})
    return contracts


def _columns_for_contract(
    context: Model6Context,
    family: str,
    fold: int,
    contract: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    source_columns = context.family_columns[family]
    source_names = context.family_names[family]
    if contract["method"] == "all":
        return source_columns.copy(), list(source_names)
    ranking_path = (
        context.config.artifact_path / "rankings" / family / f"fold_{fold}.npz"
    )
    if not ranking_path.exists():
        raise FileNotFoundError("build feature rankings first")
    with np.load(ranking_path, allow_pickle=False) as saved:
        order = saved[contract["method"]].astype(np.int32)
    count = int(contract["count"])
    if count > order.size:
        raise ValueError(f"{family} has fewer than {count} source features")
    local = order[:count]
    return source_columns[local], [source_names[int(index)] for index in local]


def _base_hyperparameters(model_type: str) -> dict[str, Any]:
    """Balanced screening defaults; later stages tune around these values."""

    if model_type == "lightgbm_gbdt":
        return {
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_data_in_leaf": 250,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.2,
            "lambda_l2": 4.0,
        }
    if model_type == "lightgbm_dart":
        return {
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_data_in_leaf": 250,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.85,
            "lambda_l2": 4.0,
            "drop_rate": 0.08,
            "skip_drop": 0.5,
        }
    if model_type == "lightgbm_goss":
        return {
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_data_in_leaf": 250,
            "feature_fraction": 0.65,
            "lambda_l1": 0.2,
            "lambda_l2": 4.0,
            "top_rate": 0.20,
            "other_rate": 0.10,
        }
    if model_type == "lightgbm_rf":
        return {
            "learning_rate": 0.10,
            "num_leaves": 63,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.45,
            "bagging_fraction": 0.70,
            "lambda_l2": 2.0,
        }
    if model_type == "xgboost":
        return {
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 20.0,
            "subsample": 0.85,
            "colsample_bytree": 0.65,
            "reg_alpha": 0.2,
            "reg_lambda": 6.0,
            "max_bin": 127,
        }
    if model_type == "catboost":
        return {
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.5,
            "rsm": 0.70,
            "subsample": 0.85,
        }
    if model_type == "random_forest":
        return {
            "n_estimators": 72,
            "max_depth": 16,
            "min_samples_leaf": 30,
            "max_features": "sqrt",
            "max_samples": 0.80,
        }
    if model_type == "extra_trees":
        return {
            "n_estimators": 96,
            "max_depth": 18,
            "min_samples_leaf": 25,
            "max_features": 0.35,
        }
    if model_type == "linear_l2":
        return {"penalty": "l2", "alpha": 1e-5, "max_iter": 45, "average": True}
    if model_type == "linear_l1":
        return {"penalty": "l1", "alpha": 1e-5, "max_iter": 45, "average": True}
    if model_type == "linear_elasticnet":
        return {
            "penalty": "elasticnet",
            "alpha": 1e-5,
            "l1_ratio": 0.20,
            "max_iter": 45,
            "average": True,
        }
    raise ValueError(f"unknown model type: {model_type}")


def hyperparameter_grid(model_type: str) -> list[dict[str, Any]]:
    """Predeclared grids for promoted learners; names make checkpoints stable."""

    base_hyper = _base_hyperparameters(model_type)

    def variants(rows: Sequence[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            {"name": name, **base_hyper, **changes}
            for name, changes in rows
        ]

    if model_type == "lightgbm_gbdt":
        return variants(
            (
                ("gbdt_balanced", {}),
                ("gbdt_shallow", {"num_leaves": 15, "min_data_in_leaf": 400}),
                ("gbdt_wide", {"num_leaves": 63, "min_data_in_leaf": 300}),
                ("gbdt_regularized", {"lambda_l1": 1.0, "lambda_l2": 10.0}),
                ("gbdt_dense_columns", {"feature_fraction": 0.90, "bagging_fraction": 0.75}),
            )
        )
    if model_type == "lightgbm_dart":
        return variants(
            (
                ("dart_balanced", {}),
                ("dart_low_drop", {"drop_rate": 0.04, "skip_drop": 0.7}),
                ("dart_high_drop", {"drop_rate": 0.15, "skip_drop": 0.35}),
                ("dart_wide", {"num_leaves": 63, "min_data_in_leaf": 350}),
            )
        )
    if model_type == "lightgbm_goss":
        return variants(
            (
                ("goss_balanced", {}),
                ("goss_conservative", {"top_rate": 0.15, "other_rate": 0.15}),
                ("goss_aggressive", {"top_rate": 0.30, "other_rate": 0.08}),
                ("goss_wide", {"num_leaves": 63, "min_data_in_leaf": 350}),
            )
        )
    if model_type == "lightgbm_rf":
        return variants(
            (
                ("lgb_rf_balanced", {}),
                ("lgb_rf_more_columns", {"feature_fraction": 0.70}),
                ("lgb_rf_deeper", {"num_leaves": 127, "min_data_in_leaf": 120}),
                ("lgb_rf_more_rows", {"bagging_fraction": 0.90}),
            )
        )
    if model_type == "xgboost":
        return variants(
            (
                ("xgb_balanced", {}),
                ("xgb_depth3", {"max_depth": 3, "min_child_weight": 10.0}),
                ("xgb_depth7", {"max_depth": 7, "min_child_weight": 35.0}),
                ("xgb_regularized", {"reg_alpha": 1.0, "reg_lambda": 12.0}),
                ("xgb_dense_columns", {"colsample_bytree": 0.90, "subsample": 0.75}),
            )
        )
    if model_type == "catboost":
        return variants(
            (
                ("cat_balanced", {}),
                ("cat_depth5", {"depth": 5, "l2_leaf_reg": 4.0}),
                ("cat_depth7", {"depth": 7, "l2_leaf_reg": 10.0}),
                ("cat_low_random", {"random_strength": 0.1, "rsm": 0.85}),
                ("cat_regularized", {"l2_leaf_reg": 15.0, "random_strength": 1.0}),
            )
        )
    if model_type == "random_forest":
        return variants(
            (
                ("rf_balanced", {}),
                ("rf_shallow", {"max_depth": 12, "min_samples_leaf": 50}),
                ("rf_deep", {"max_depth": None, "min_samples_leaf": 50}),
                # Deliberate 20%-column stress test.  Successive halving keeps
                # this to one fold unless the large CPU cost buys real TSROC.
                ("rf_more_columns", {"max_features": 0.20, "min_samples_leaf": 40}),
            )
        )
    if model_type == "extra_trees":
        return variants(
            (
                ("extra_balanced", {}),
                (
                    "extra_shallow",
                    {
                        "max_depth": 12,
                        "min_samples_leaf": 40,
                    },
                ),
                (
                    "extra_deep",
                    {
                        "max_depth": None,
                        "min_samples_leaf": 50,
                    },
                ),
                ("extra_sparse_columns", {"max_features": "sqrt", "min_samples_leaf": 20}),
            )
        )
    if model_type in {"linear_l2", "linear_l1"}:
        return variants(
            tuple(
                (f"{model_type}_a{str(alpha).replace('.', 'p')}", {"alpha": alpha})
                for alpha in (3e-6, 1e-5, 3e-5, 1e-4)
            )
        )
    if model_type == "linear_elasticnet":
        return variants(
            (
                ("elastic_a3e6_l10", {"alpha": 3e-6, "l1_ratio": 0.10}),
                ("elastic_a1e5_l20", {"alpha": 1e-5, "l1_ratio": 0.20}),
                ("elastic_a3e5_l50", {"alpha": 3e-5, "l1_ratio": 0.50}),
                ("elastic_a1e4_l80", {"alpha": 1e-4, "l1_ratio": 0.80}),
            )
        )
    raise ValueError(f"unknown model type: {model_type}")


def _fit_model(
    *,
    model_type: str,
    hyperparameters: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_weight: np.ndarray,
    x_stop: np.ndarray,
    y_stop: np.ndarray,
    stop_weight: np.ndarray,
    stop_time: np.ndarray,
    rounds: int,
    patience: int,
    seed: int,
    threads: int,
    device: str,
) -> FittedModel:
    """Fit one learner using a separate early-stop role when supported."""

    parameters = {
        key: value for key, value in hyperparameters.items() if key != "name"
    }
    if model_type.startswith("lightgbm_"):
        boosting = model_type.removeprefix("lightgbm_")
        lgb_parameters: dict[str, Any] = {
            "objective": "binary",
            "metric": "None",
            "boosting_type": boosting,
            "verbosity": -1,
            "num_threads": threads,
            "seed": seed,
            "feature_fraction_seed": seed + 1,
            "bagging_seed": seed + 2,
            "data_random_seed": seed + 3,
            "deterministic": device == "cpu",
            "force_col_wise": device == "cpu",
            **parameters,
        }
        if boosting in {"gbdt", "dart", "rf"}:
            lgb_parameters["bagging_freq"] = 1
        if boosting == "goss":
            lgb_parameters.pop("bagging_fraction", None)
            lgb_parameters.pop("bagging_freq", None)
        if device == "cuda":
            lgb_parameters.update({"device_type": "cuda", "max_bin": 63})
        train_set = lgb.Dataset(x_train, label=y_train, weight=train_weight)
        stop_set = lgb.Dataset(
            x_stop,
            label=y_stop,
            weight=stop_weight,
            reference=train_set,
        )
        callbacks: list[Any] = [lgb.log_evaluation(period=0)]
        # LightGBM explicitly disables early stopping for DART.
        if boosting != "dart":
            callbacks.append(
                lgb.early_stopping(patience, first_metric_only=True, verbose=False)
            )

        def evaluator(prediction: np.ndarray, dataset: lgb.Dataset):
            return "tsroc", tsauc(dataset.get_label(), prediction, stop_time), True

        model = lgb.train(
            lgb_parameters,
            train_set,
            num_boost_round=rounds,
            valid_sets=[stop_set],
            feval=evaluator,
            callbacks=callbacks,
        )
        best = int(model.best_iteration or rounds)
        return FittedModel(model_type=model_type, model=model, best_iteration=best)

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            n_estimators=rounds,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=threads,
            device=device,
            verbosity=0,
            **parameters,
        )
        model.fit(x_train, y_train, sample_weight=train_weight, verbose=False)
        return FittedModel(model_type=model_type, model=model, best_iteration=rounds)

    if model_type == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=rounds,
            random_seed=seed,
            thread_count=threads,
            task_type="GPU" if device == "cuda" else "CPU",
            allow_writing_files=False,
            bootstrap_type="Bernoulli",
            verbose=False,
            **parameters,
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=train_weight,
            eval_set=(x_stop, y_stop),
            early_stopping_rounds=patience,
            use_best_model=True,
            verbose=False,
        )
        best = int(model.get_best_iteration()) + 1
        return FittedModel(model_type=model_type, model=model, best_iteration=max(1, best))

    if model_type in {"random_forest", "extra_trees"}:
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier as Classifier
        else:
            from sklearn.ensemble import ExtraTreesClassifier as Classifier
        model = Classifier(
            random_state=seed,
            n_jobs=threads,
            criterion="gini",
            **parameters,
        )
        model.fit(x_train, y_train, sample_weight=train_weight)
        return FittedModel(
            model_type=model_type,
            model=model,
            best_iteration=int(parameters.get("n_estimators", 0)),
        )

    if model_type.startswith("linear_"):
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(copy=True)
        scaler.fit(x_train, sample_weight=train_weight)
        transformed = scaler.transform(x_train).astype(np.float32, copy=False)
        np.clip(transformed, -25.0, 25.0, out=transformed)
        model = SGDClassifier(
            loss="log_loss",
            learning_rate="optimal",
            tol=1e-4,
            random_state=seed,
            **parameters,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(transformed, y_train, sample_weight=train_weight)
        return FittedModel(
            model_type=model_type,
            model=model,
            scaler=scaler,
            best_iteration=int(parameters.get("max_iter", 0)),
        )
    raise ValueError(f"unknown model type: {model_type}")


def _predict_array(fitted: FittedModel, matrix: np.ndarray) -> np.ndarray:
    if fitted.model_type.startswith("linear_"):
        transformed = fitted.scaler.transform(matrix).astype(np.float32, copy=False)
        np.clip(transformed, -25.0, 25.0, out=transformed)
        prediction = _logistic(fitted.model.decision_function(transformed))
    elif fitted.model_type.startswith("lightgbm_"):
        prediction = fitted.model.predict(matrix, num_iteration=fitted.best_iteration)
    else:
        prediction = fitted.model.predict_proba(matrix)[:, 1]
    return np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1 - 1e-7)


def _predict_feature_rows(
    context: Model6Context,
    fitted: FittedModel,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    prediction = np.empty(rows.size, dtype=np.float32)
    chunk = context.config.prediction_chunk_rows
    for left in range(0, rows.size, chunk):
        right = min(left + chunk, rows.size)
        matrix = _sanitize(
            matrix_rows(context.data.joint_matrix, rows[left:right], columns)
        )
        prediction[left:right] = _predict_array(fitted, matrix).astype(np.float32)
        del matrix
    return prediction


def _column_hash(columns: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(columns, dtype=np.int32).tobytes()).hexdigest()


def _fit_and_score_feature_candidate(
    context: Model6Context,
    *,
    family: str,
    fold: int,
    feature_contract: dict[str, Any],
    model_type: str,
    hyperparameters: dict[str, Any],
    train_cap: int,
    stop_cap: int,
    rounds: int,
    patience: int,
    result_path: Path,
    save_model: bool = False,
) -> dict[str, Any]:
    """Fit on one role, stop on another, and score every held-fold row."""

    hyper_name = str(hyperparameters.get("name", "screening_default"))
    run_label = (
        f"{family}/{feature_contract['key']}/{model_type}/"
        f"{hyper_name}/fold{fold}"
    )
    columns, names = _columns_for_contract(
        context, family, fold, feature_contract
    )
    train_mask, stop_mask, score_mask, stop_fold = base.development_fold_roles(
        context.split, fold, 5
    )
    stage_seed = int(
        context.config.seed
        + 10_000 * fold
        + int(hashlib.sha256(str(result_path).encode()).hexdigest()[:8], 16) % 100_000
    )
    train_rows = _cap_rows(
        np.flatnonzero(train_mask & context.split.sampled_row_mask),
        train_cap,
        stage_seed + 1,
    )
    stop_rows = _cap_rows(
        np.flatnonzero(stop_mask & context.split.sampled_row_mask),
        stop_cap,
        stage_seed + 2,
    )
    score_rows = np.flatnonzero(score_mask)
    contract = {
        "experiment_version": MODEL6_VERSION,
        "family": family,
        "score_fold": fold,
        "early_stop_fold": stop_fold,
        "feature_contract": feature_contract,
        "feature_count": int(columns.size),
        "column_hash": _column_hash(columns),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "train_cap": train_cap,
        "stop_cap": stop_cap,
        "rounds": rounds,
        "patience": patience,
        "seed": stage_seed,
        "actual_device": context.actual_device,
        "score_rows_are_complete_held_series": True,
        "outer_holdout_read": False,
    }
    contract_hash = _hash_json(contract)
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("contract_hash") != contract_hash:
            raise ValueError(f"stale fit checkpoint: {result_path}")
        prediction_path = result_path.with_suffix(".prediction.f32.npy")
        model_path = result_path.with_suffix(".joblib")
        if save_model and (not prediction_path.exists() or not model_path.exists()):
            raise ValueError(f"incomplete final-fit checkpoint: {result_path}")
        print(
            f"{run_label}: checkpoint TSROC={saved['tsroc']:.6f}",
            flush=True,
        )
        return saved

    print(
        f"{run_label}: {train_rows.size:,} train, {stop_rows.size:,} stop, "
        f"{score_rows.size:,} score, {columns.size:,} features",
        flush=True,
    )
    started = time.perf_counter()
    x_train = _sanitize(matrix_rows(context.data.joint_matrix, train_rows, columns))
    x_stop = _sanitize(matrix_rows(context.data.joint_matrix, stop_rows, columns))
    y_train = context.data.metadata["target"][train_rows]
    y_stop = context.data.metadata["target"][stop_rows]
    train_weight = pair_weights_from_reference(
        y_train,
        context.data.metadata["time_online"][train_rows],
        context.data.metadata["target"][train_mask],
        context.data.metadata["time_online"][train_mask],
    )
    stop_weight = pair_weights_from_reference(
        y_stop,
        context.data.metadata["time_online"][stop_rows],
        context.data.metadata["target"][stop_mask],
        context.data.metadata["time_online"][stop_mask],
    )
    fitted = _fit_model(
        model_type=model_type,
        hyperparameters=hyperparameters,
        x_train=x_train,
        y_train=y_train,
        train_weight=train_weight,
        x_stop=x_stop,
        y_stop=y_stop,
        stop_weight=stop_weight,
        stop_time=context.data.metadata["time_online"][stop_rows],
        rounds=rounds,
        patience=patience,
        seed=stage_seed,
        threads=context.config.num_threads,
        device=context.actual_device,
    )
    del x_train, x_stop, y_train, y_stop, train_weight, stop_weight
    gc.collect()
    prediction = _predict_feature_rows(context, fitted, score_rows, columns)
    target = context.data.metadata["target"][score_rows]
    record = {
        **contract,
        "contract_hash": contract_hash,
        "tsroc": float(
            tsauc(
                target,
                prediction,
                context.data.metadata["time_online"][score_rows],
            )
        ),
        "ordinary_roc_diagnostic": float(roc_auc_score(target, prediction)),
        "best_iteration": int(fitted.best_iteration),
        "train_rows": int(train_rows.size),
        "stop_rows": int(stop_rows.size),
        "score_rows": int(score_rows.size),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if save_model:
        np.save(result_path.with_suffix(".prediction.f32.npy"), prediction)
        np.savez_compressed(
            result_path.with_suffix(".columns.npz"),
            columns=columns.astype(np.int32),
            names=np.asarray(names, dtype=np.str_),
        )
        joblib.dump(fitted, result_path.with_suffix(".joblib"), compress=0)
    _atomic_json(result_path, record)
    print(
        f"{run_label}: TSROC={record['tsroc']:.6f}, "
        f"{record['elapsed_seconds']/60:.2f} min",
        flush=True,
    )
    del fitted, prediction, target
    gc.collect()
    return record


def _candidate_summary(
    rows: Sequence[dict[str, Any]], key_fields: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, records in grouped.items():
        scores = [record["tsroc"] for record in sorted(records, key=lambda item: item["score_fold"])]
        output.append(
            {
                **dict(zip(key_fields, key)),
                **_mean_se(scores),
                "folds": [record["score_fold"] for record in records],
                "feature_count": records[0]["feature_count"],
            }
        )
    return sorted(output, key=lambda row: (-row["fold_mean"], str(row)))


def _promote_feature_contracts(
    ranked: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Keep strong candidates while retaining ranking-method diversity."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        key = row["feature_key"]
        if key not in seen and len(selected) < maximum:
            selected.append(
                {
                    "method": row["feature_method"],
                    "count": row["feature_count_label"],
                    "key": key,
                    "screen_fold_mean": row["fold_mean"],
                }
            )
            seen.add(key)

    # Three merit slots retain the outright leaders; the remaining slots stop
    # one ranking family from crowding out strong, complementary alternatives.
    for row in ranked[: max(2, maximum // 2)]:
        add(row)
    # Keep the full family even when the reference learner ranks it poorly:
    # linear and randomized-tree models can react differently to wide inputs,
    # and the user explicitly requested an all-feature comparison.
    for method in ("all", "consensus", "gain_corr", "gain", "split", "fisher"):
        champion = next((row for row in ranked if row["feature_method"] == method), None)
        if champion is not None:
            add(champion)
    for row in ranked:
        add(row)
    return selected


def run_feature_search(context: Model6Context) -> dict[str, Any]:
    """Test every requested feature method/count with a common reference model."""

    build_feature_rankings(context)
    config = context.config
    report: dict[str, Any] = {
        "stage": "feature_search",
        "selection_metric": "TSROC",
        "outer_holdout_read": False,
        "families": {},
    }
    for family in FAMILIES:
        records: list[dict[str, Any]] = []
        for feature_contract in feature_contracts(config):
            for fold in config.feature_search_folds:
                path = (
                    config.artifact_path
                    / "feature_search"
                    / family
                    / feature_contract["key"]
                    / f"fold_{fold}.json"
                )
                fit = _fit_and_score_feature_candidate(
                    context,
                    family=family,
                    fold=fold,
                    feature_contract=feature_contract,
                    model_type="lightgbm_gbdt",
                    hyperparameters=_base_hyperparameters("lightgbm_gbdt"),
                    train_cap=config.feature_search_train_rows,
                    stop_cap=config.feature_search_stop_rows,
                    rounds=config.feature_search_rounds,
                    patience=config.feature_search_patience,
                    result_path=path,
                )
                records.append(
                    {
                        **fit,
                        "feature_key": feature_contract["key"],
                        "feature_method": feature_contract["method"],
                        "feature_count_label": feature_contract["count"],
                    }
                )
        ranking = _candidate_summary(
            records,
            ("feature_key", "feature_method", "feature_count_label"),
        )
        promoted = _promote_feature_contracts(
            ranking, config.promoted_feature_contracts
        )
        report["families"][family] = {
            "ranking": ranking,
            "promoted": promoted,
        }
    _atomic_json(config.artifact_path / "feature_search/summary.json", report)
    return report


def run_model_search(context: Model6Context) -> dict[str, Any]:
    """Compare every requested learner on Stage-1-promoted feature sets."""

    feature_path = context.config.artifact_path / "feature_search/summary.json"
    if not feature_path.exists():
        run_feature_search(context)
    feature_report = json.loads(feature_path.read_text(encoding="utf-8"))
    config = context.config
    report: dict[str, Any] = {
        "stage": "model_search",
        "selection_metric": "TSROC",
        "linear_note": (
            "SGD logistic L2 is the scalable classification analogue of ridge; "
            "L1 is the analogue of lasso; elastic-net mixes both."
        ),
        "linear_successive_halving": (
            "core linear penalties stop repeating after four promoted feature "
            "contracts; candidate penalties stop after the leading contract"
        ),
        "outer_holdout_read": False,
        "families": {},
    }
    for family in FAMILIES:
        records: list[dict[str, Any]] = []
        promoted_features = feature_report["families"][family]["promoted"]
        for feature_index, feature_contract in enumerate(promoted_features):
            model_types = list(config.model_types)
            # The linear penalties are deliberately broad-tested, then
            # successively halved after repeated large deficits. Core has
            # Fisher, consensus, and all-feature evidence; candidate keeps one
            # complete leading-contract comparison before pruning repetition.
            if (family == "core" and feature_index >= 4) or (
                family == "candidate" and feature_index >= 1
            ):
                model_types = [
                    model_type
                    for model_type in model_types
                    if not model_type.startswith("linear_")
                ]
            for model_type in model_types:
                for fold in config.model_search_folds:
                    path = (
                        config.artifact_path
                        / "model_search"
                        / family
                        / feature_contract["key"]
                        / model_type
                        / f"fold_{fold}.json"
                    )
                    fit = _fit_and_score_feature_candidate(
                        context,
                        family=family,
                        fold=fold,
                        feature_contract=feature_contract,
                        model_type=model_type,
                        hyperparameters=_base_hyperparameters(model_type),
                        train_cap=config.model_search_train_rows,
                        stop_cap=config.model_search_stop_rows,
                        rounds=config.model_search_rounds,
                        patience=config.model_search_patience,
                        result_path=path,
                    )
                    records.append(
                        {
                            **fit,
                            "feature_key": feature_contract["key"],
                            "feature_method": feature_contract["method"],
                            "feature_count_label": feature_contract["count"],
                            "model_type": model_type,
                        }
                    )
        ranking = _candidate_summary(
            records,
            (
                "model_type",
                "feature_key",
                "feature_method",
                "feature_count_label",
            ),
        )
        best_by_model = []
        for model_type in config.model_types:
            candidates = [row for row in ranking if row["model_type"] == model_type]
            if not candidates:
                raise AssertionError(f"no model-search result for {family}/{model_type}")
            best_by_model.append(candidates[0])
        best_by_model.sort(key=lambda row: -row["fold_mean"])
        promoted_models = best_by_model[: config.promoted_model_types]
        report["families"][family] = {
            "ranking": ranking,
            "best_feature_contract_by_model": best_by_model,
            "promoted_models": promoted_models,
        }
    _atomic_json(config.artifact_path / "model_search/summary.json", report)
    return report


def _postprocess_oof(
    context: Model6Context, prediction: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Nested-select causal persistence without using a fold on itself.

    The grid contains the raw prediction, the original additive logit-space
    recurrence, and the absorbing-target probability recurrence requested for
    this refresh.  For held fold ``h``, the winning transform is chosen using
    only the other four development folds.
    """

    development_rows = context.split.development_rows
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    owner = context.data.metadata["series_position"][development_rows]
    fold = context.split.row_inner_fold[development_rows]
    candidates: dict[str, np.ndarray] = {
        "raw": np.asarray(prediction, dtype=np.float64).copy()
    }
    transform_specs: dict[str, dict[str, Any]] = {
        "raw": {"kind": "raw"}
    }
    for penalty in context.config.causal_penalties:
        if penalty is None:
            continue
        name = f"logit_decayed_max_{penalty:g}"
        # causal_decayed_max works in margin space. Convert back to a bounded
        # probability so downstream logit ensembling has a stable contract.
        candidates[name] = _logistic(
            causal_decayed_max(prediction, owner, penalty)
        )
        transform_specs[name] = {
            "kind": "logit_subtractive_decayed_max",
            "penalty": float(penalty),
        }
    for factor in context.config.probability_decay_factors:
        name = f"probability_decayed_max_{factor:g}"
        candidates[name] = causal_probability_decayed_max(
            prediction, owner, factor
        )
        transform_specs[name] = {
            "kind": "probability_multiplicative_decayed_max",
            "decay_factor": float(factor),
        }
    fold_scores = {
        name: [
            float(tsauc(target[fold == held], values[fold == held], online_time[fold == held]))
            for held in range(5)
        ]
        for name, values in candidates.items()
    }
    nested = np.empty(prediction.size, dtype=np.float32)
    nested_names: list[str] = []
    nested_scores: list[float] = []
    for held in range(5):
        name = max(
            candidates,
            key=lambda candidate: (
                float(np.mean(np.delete(fold_scores[candidate], held))),
                candidate == "raw",
            ),
        )
        mask = fold == held
        nested[mask] = candidates[name][mask]
        nested_names.append(name)
        nested_scores.append(float(fold_scores[name][held]))
    deployment_name = max(
        candidates,
        key=lambda candidate: (
            float(np.mean(fold_scores[candidate])),
            candidate == "raw",
        ),
    )
    report = {
        "method": (
            "nested raw/additive-logit/multiplicative-probability "
            "causal decayed-max selection"
        ),
        "fold_grid_tsroc": fold_scores,
        "candidate_transform_specs": transform_specs,
        "selected_without_held_fold": nested_names,
        "nested_tsroc": {
            **_mean_se(nested_scores),
            "pooled_development_oof": float(tsauc(target, nested, online_time)),
        },
        "deployment_transform": deployment_name,
        "deployment_transform_spec": transform_specs[deployment_name],
        # Retain scalar compatibility fields for reports written by Model 6 v1.
        "deployment_penalty": transform_specs[deployment_name].get("penalty"),
        "deployment_decay_factor": transform_specs[deployment_name].get(
            "decay_factor"
        ),
        "deployment_fold_mean_tsroc": float(np.mean(fold_scores[deployment_name])),
    }
    return nested, report


def _detector_oof_matrix(
    context: Model6Context, family: str
) -> tuple[np.ndarray, list[str], dict[str, list[float]]]:
    banks = list(context.family_banks[family])
    arrays = []
    scores: dict[str, list[float]] = {}
    for bank in banks:
        directory = context.source_config.artifact_path / "screen" / family / bank
        path = directory / "development_oof.f32.npy"
        summary_path = directory / "summary.json"
        if not path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Model 5 detector OOF checkpoint is missing: {path}")
        arrays.append(np.load(path, mmap_mode="r"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scores[bank] = [float(value) for value in summary["tsauc_diagnostic"]["fold_scores"]]
    matrix = np.column_stack(arrays).astype(np.float32)
    if matrix.shape[0] != context.split.development_rows.size:
        raise ValueError("detector OOF rows do not align with Model 6 development rows")
    return matrix, banks, scores


def _stack_features(probabilities: np.ndarray) -> np.ndarray:
    """Detector logits plus simple agreement/disagreement meta-features."""

    margins = _safe_logit(probabilities).astype(np.float32)
    ordered = np.sort(margins, axis=1)
    extras = np.column_stack(
        (
            margins.mean(axis=1),
            margins.std(axis=1),
            margins.min(axis=1),
            margins.max(axis=1),
            np.median(margins, axis=1),
            ordered[:, -1] - ordered[:, -2] if margins.shape[1] > 1 else margins[:, 0],
        )
    ).astype(np.float32)
    return np.ascontiguousarray(np.column_stack((margins, extras)), dtype=np.float32)


def _score_oof_component(
    context: Model6Context,
    family: str,
    name: str,
    raw_prediction: np.ndarray,
    detail: dict[str, Any],
) -> dict[str, Any]:
    development_rows = context.split.development_rows
    fold = context.split.row_inner_fold[development_rows]
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    raw_scores = [
        float(
            tsauc(
                target[fold == held],
                raw_prediction[fold == held],
                online_time[fold == held],
            )
        )
        for held in range(5)
    ]
    nested_prediction, postprocess = _postprocess_oof(context, raw_prediction)
    directory = context.config.artifact_path / "detector_stack" / family / name
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "raw_oof.f32.npy", raw_prediction.astype(np.float32))
    np.save(directory / "nested_oof.f32.npy", nested_prediction.astype(np.float32))
    report = {
        "family": family,
        "component": name,
        "first_level_predictions_are_model5_oof": True,
        "outer_holdout_read": False,
        "raw_tsroc": {
            **_mean_se(raw_scores),
            "pooled_development_oof": float(tsauc(target, raw_prediction, online_time)),
        },
        "postprocess": postprocess,
        "ensemble_prediction_path": str(
            (directory / "nested_oof.f32.npy").relative_to(context.config.project_root)
        ),
        **detail,
    }
    _atomic_json(directory / "summary.json", report)
    return report


def _nested_top_n_detector_prediction(
    context: Model6Context,
    probabilities: np.ndarray,
    banks: list[str],
    bank_fold_scores: dict[str, list[float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rank banks and choose ensemble width without using the held fold."""

    development_rows = context.split.development_rows
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    fold = context.split.row_inner_fold[development_rows]
    margin = _safe_logit(probabilities)
    grid = sorted(
        set(
            count
            for count in (1, 2, 3, 4, 5, 7, 10, len(banks))
            if count <= len(banks)
        )
    )
    nested = np.empty(target.size, dtype=np.float32)
    held_records = []
    for held in range(5):
        order = sorted(
            range(len(banks)),
            key=lambda index: (
                -float(np.mean(np.delete(bank_fold_scores[banks[index]], held))),
                banks[index],
            ),
        )
        candidate_prediction = {
            count: _logistic(margin[:, order[:count]].mean(axis=1))
            for count in grid
        }
        tuning_score = {
            count: float(
                np.mean(
                    [
                        tsauc(
                            target[fold == other],
                            candidate_prediction[count][fold == other],
                            online_time[fold == other],
                        )
                        for other in range(5)
                        if other != held
                    ]
                )
            )
            for count in grid
        }
        selected_count = max(grid, key=lambda count: (tuning_score[count], -count))
        mask = fold == held
        nested[mask] = candidate_prediction[selected_count][mask]
        held_records.append(
            {
                "held_fold": held,
                "bank_order_without_held": [banks[index] for index in order],
                "selected_count_without_held": selected_count,
                "tuning_mean_tsroc": tuning_score,
            }
        )

    deployment_order = sorted(
        range(len(banks)),
        key=lambda index: (-float(np.mean(bank_fold_scores[banks[index]])), banks[index]),
    )
    deployment_grid = {}
    for count in grid:
        prediction = _logistic(margin[:, deployment_order[:count]].mean(axis=1))
        deployment_grid[count] = float(
            np.mean(
                [
                    tsauc(
                        target[fold == held],
                        prediction[fold == held],
                        online_time[fold == held],
                    )
                    for held in range(5)
                ]
            )
        )
    deployment_count = max(grid, key=lambda count: (deployment_grid[count], -count))
    return nested, {
        "method": "nested top-N logit mean",
        "held_fold_selection": held_records,
        "deployment_bank_order": [banks[index] for index in deployment_order],
        "deployment_count": deployment_count,
        "deployment_grid_mean_tsroc": deployment_grid,
    }


def _cross_fit_detector_meta_model(
    context: Model6Context,
    family: str,
    stack_matrix: np.ndarray,
    model_type: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a second level only on first-level OOF rows outside its score fold."""

    config = context.config
    development_rows = context.split.development_rows
    fold = context.split.row_inner_fold[development_rows]
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    oof = np.empty(target.size, dtype=np.float32)
    records = []
    hyper = _base_hyperparameters(model_type)
    for held in range(5):
        directory = config.artifact_path / "detector_stack" / family / model_type
        result_path = directory / f"fold_{held}.json"
        prediction_path = directory / f"fold_{held}.prediction.f32.npy"
        stop_fold = (held + 1) % 5
        score_mask = fold == held
        stop_mask = fold == stop_fold
        train_mask = ~(score_mask | stop_mask)
        train_positions = _cap_rows(
            np.flatnonzero(train_mask),
            config.detector_stack_train_rows,
            config.seed + 70_000 + 101 * held,
        )
        stop_positions = _cap_rows(
            np.flatnonzero(stop_mask),
            min(config.detector_stack_train_rows // 2, int(stop_mask.sum())),
            config.seed + 71_000 + 101 * held,
        )
        score_positions = np.flatnonzero(score_mask)
        contract = {
            "stage": "detector_oof_meta",
            "family": family,
            "model_type": model_type,
            "held_fold": held,
            "stop_fold": stop_fold,
            "banks": list(context.family_banks[family]),
            "meta_feature_count": int(stack_matrix.shape[1]),
            "hyperparameters": hyper,
            "train_cap": config.detector_stack_train_rows,
            "first_level_oof_only": True,
            "outer_holdout_read": False,
        }
        contract_hash = _hash_json(contract)
        if result_path.exists() and prediction_path.exists():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if record.get("contract_hash") != contract_hash:
                raise ValueError(f"stale detector-stack checkpoint: {result_path}")
            prediction = np.load(prediction_path, allow_pickle=False)
            print(
                f"detector stack {family}/{model_type}/fold{held}: "
                f"checkpoint TSROC={record['tsroc']:.6f}",
                flush=True,
            )
        else:
            started = time.perf_counter()
            train_weight = pair_weights_from_reference(
                target[train_positions],
                online_time[train_positions],
                target[train_mask],
                online_time[train_mask],
            )
            stop_weight = pair_weights_from_reference(
                target[stop_positions],
                online_time[stop_positions],
                target[stop_mask],
                online_time[stop_mask],
            )
            fitted = _fit_model(
                model_type=model_type,
                hyperparameters=hyper,
                x_train=stack_matrix[train_positions],
                y_train=target[train_positions],
                train_weight=train_weight,
                x_stop=stack_matrix[stop_positions],
                y_stop=target[stop_positions],
                stop_weight=stop_weight,
                stop_time=online_time[stop_positions],
                rounds=config.model_search_rounds,
                patience=config.model_search_patience,
                seed=config.seed + 72_000 + held,
                threads=config.num_threads,
                device=context.actual_device,
            )
            prediction = _predict_array(fitted, stack_matrix[score_positions]).astype(
                np.float32
            )
            record = {
                **contract,
                "contract_hash": contract_hash,
                "tsroc": float(
                    tsauc(target[score_positions], prediction, online_time[score_positions])
                ),
                "best_iteration": fitted.best_iteration,
                "train_rows": int(train_positions.size),
                "stop_rows": int(stop_positions.size),
                "score_rows": int(score_positions.size),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            directory.mkdir(parents=True, exist_ok=True)
            np.save(prediction_path, prediction)
            _atomic_json(result_path, record)
            print(
                f"detector stack {family}/{model_type}/fold{held}: "
                f"TSROC={record['tsroc']:.6f}",
                flush=True,
            )
            del fitted, train_weight, stop_weight
            gc.collect()
        if prediction.shape != (score_positions.size,):
            raise ValueError(f"bad detector stack prediction: {prediction_path}")
        oof[score_positions] = prediction
        records.append(record)
    return oof, {"cross_fit_records": records, "meta_model": model_type}


def _nested_greedy_detector_prediction(
    context: Model6Context,
    probabilities: np.ndarray,
    banks: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Directly optimize unequal bank weights without using the held fold."""

    development_rows = context.split.development_rows
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    fold = context.split.row_inner_fold[development_rows]
    bank_tuning_series_per_fold = 100
    tuning_rows = _ensemble_tuning_mask(
        context, series_per_fold=bank_tuning_series_per_fold
    )
    # Bank-level weighting is a diversity supplement, not the primary model
    # grid. A coarser exact-TSROC search avoids starving the multi-model stage
    # on machines with many logical but few high-throughput CPU cores.
    bank_config = replace(
        context.config,
        ensemble_weight_step=0.10,
        ensemble_max_components=5,
    )
    margins = {
        bank: _safe_logit(probabilities[:, index])
        for index, bank in enumerate(banks)
    }
    nested = np.empty(target.size, dtype=np.float32)
    held_recipes = []
    for held in range(5):
        recipe = _greedy_recipe(
            target,
            online_time,
            margins,
            tuning_rows & (fold != held),
            bank_config,
        )
        prediction = _apply_recipe(margins, recipe)
        nested[fold == held] = prediction[fold == held]
        held_recipes.append({"held_fold": held, "recipe": recipe})
    deployment_recipe = _greedy_recipe(
        target, online_time, margins, tuning_rows, bank_config
    )
    return nested, {
        "method": "nested greedy unequal logit weights over detector banks",
        "held_recipes": held_recipes,
        "deployment_recipe": deployment_recipe,
        "weight_tuning_series_per_fold": bank_tuning_series_per_fold,
        "weight_step": bank_config.ensemble_weight_step,
        "maximum_components": bank_config.ensemble_max_components,
    }


def run_detector_stack(context: Model6Context) -> dict[str, Any]:
    """Build cheap, diverse family models from selected detector OOF scores."""

    report: dict[str, Any] = {
        "stage": "detector_stack",
        "selection_metric": "TSROC",
        "first_level_source": "Model 5 OOF predictions",
        "outer_holdout_read": False,
        "families": {},
    }
    for family in FAMILIES:
        probabilities, banks, bank_scores = _detector_oof_matrix(context, family)
        components = []
        logit_mean = _logistic(_safe_logit(probabilities).mean(axis=1)).astype(np.float32)
        components.append(
            _score_oof_component(
                context,
                family,
                "logit_mean",
                logit_mean,
                {"banks": banks, "deployment": {"method": "all-bank logit mean"}},
            )
        )
        top_n, top_n_detail = _nested_top_n_detector_prediction(
            context, probabilities, banks, bank_scores
        )
        components.append(
            _score_oof_component(
                context, family, "nested_top_n", top_n, top_n_detail
            )
        )
        greedy, greedy_detail = _nested_greedy_detector_prediction(
            context, probabilities, banks
        )
        components.append(
            _score_oof_component(
                context, family, "greedy_banks", greedy, greedy_detail
            )
        )
        stack_matrix = _stack_features(probabilities)
        for model_type in STACK_MODEL_TYPES:
            prediction, detail = _cross_fit_detector_meta_model(
                context, family, stack_matrix, model_type
            )
            components.append(
                _score_oof_component(
                    context, family, f"meta_{model_type}", prediction, detail
                )
            )
        report["families"][family] = {
            "banks": banks,
            "bank_count": len(banks),
            "components": sorted(
                components,
                key=lambda row: -row["postprocess"]["nested_tsroc"]["fold_mean"],
            ),
        }
        del probabilities, stack_matrix
        gc.collect()
    _atomic_json(context.config.artifact_path / "detector_stack/summary.json", report)
    return report


def run_hyperparameter_tuning(context: Model6Context) -> dict[str, Any]:
    """Tune promoted pairs with checkpointed, fold-wise successive halving.

    Every configuration is evaluated on the first tuning fold.  The best half
    (with at least two survivors) advances to the next fold.  Only candidates
    that survive through every tuning fold can win their model-family slot.
    This avoids comparing a lucky one-fold result with a complete estimate.
    """

    model_path = context.config.artifact_path / "model_search/summary.json"
    if not model_path.exists():
        run_model_search(context)
    model_report = json.loads(model_path.read_text(encoding="utf-8"))
    config = context.config
    report: dict[str, Any] = {
        "stage": "hyperparameter_tuning",
        "selection_metric": "TSROC",
        "successive_halving": {
            "unit": "complete held-out series fold",
            "rule": "after each non-final fold keep ceil(active/2), minimum 2",
            "winner_requirement": "must survive and score every tuning fold",
            "fold_order": list(config.tuning_folds),
        },
        "outer_holdout_read": False,
        "families": {},
    }
    for family in FAMILIES:
        records: list[dict[str, Any]] = []
        complete_records: list[dict[str, Any]] = []
        halving_history: list[dict[str, Any]] = []
        for promoted in model_report["families"][family]["promoted_models"]:
            model_type = promoted["model_type"]
            feature_contract = {
                "method": promoted["feature_method"],
                "count": promoted["feature_count_label"],
                "key": promoted["feature_key"],
            }
            grid = hyperparameter_grid(model_type)
            hyper_by_name = {str(hyper["name"]): hyper for hyper in grid}
            active_names = list(hyper_by_name)
            candidate_records: dict[str, list[dict[str, Any]]] = {
                name: [] for name in active_names
            }
            model_history: list[dict[str, Any]] = []
            for rung, fold in enumerate(config.tuning_folds):
                entered_names = list(active_names)
                for hyper_name in entered_names:
                    hyper = hyper_by_name[hyper_name]
                    result_path = (
                        config.artifact_path
                        / "tuning"
                        / family
                        / model_type
                        / feature_contract["key"]
                        / hyper_name
                        / f"fold_{fold}.json"
                    )
                    fit = _fit_and_score_feature_candidate(
                        context,
                        family=family,
                        fold=fold,
                        feature_contract=feature_contract,
                        model_type=model_type,
                        hyperparameters=hyper,
                        train_cap=config.tuning_train_rows,
                        stop_cap=config.tuning_stop_rows,
                        rounds=config.tuning_rounds,
                        patience=config.tuning_patience,
                        result_path=result_path,
                    )
                    records.append(
                        {
                            **fit,
                            "feature_key": feature_contract["key"],
                            "feature_method": feature_contract["method"],
                            "feature_count_label": feature_contract["count"],
                            "model_type": model_type,
                            "hyper_name": hyper_name,
                        }
                    )
                    candidate_records[hyper_name].append(records[-1])

                ranked_names = sorted(
                    entered_names,
                    key=lambda name: (
                        -float(
                            np.mean(
                                [row["tsroc"] for row in candidate_records[name]]
                            )
                        ),
                        name,
                    ),
                )
                is_final_rung = rung == len(config.tuning_folds) - 1
                if is_final_rung:
                    active_names = ranked_names
                else:
                    survivor_count = min(
                        len(ranked_names), max(2, math.ceil(len(ranked_names) / 2))
                    )
                    active_names = ranked_names[:survivor_count]
                model_history.append(
                    {
                        "rung": rung,
                        "score_fold": fold,
                        "entered": entered_names,
                        "ranked": [
                            {
                                "hyper_name": name,
                                "mean_tsroc_to_date": float(
                                    np.mean(
                                        [
                                            row["tsroc"]
                                            for row in candidate_records[name]
                                        ]
                                    )
                                ),
                                "fold_scores": [
                                    float(row["tsroc"])
                                    for row in candidate_records[name]
                                ],
                            }
                            for name in ranked_names
                        ],
                        "advanced": list(active_names),
                    }
                )
            for hyper_name in active_names:
                if len(candidate_records[hyper_name]) != len(config.tuning_folds):
                    raise AssertionError("a tuning finalist lacks a complete fold set")
                complete_records.extend(candidate_records[hyper_name])
            halving_history.append(
                {
                    "model_type": model_type,
                    "feature_key": feature_contract["key"],
                    "initial_configuration_count": len(grid),
                    "complete_configurations": list(active_names),
                    "rungs": model_history,
                }
            )

        ranking = _candidate_summary(
            complete_records,
            (
                "model_type",
                "feature_key",
                "feature_method",
                "feature_count_label",
                "hyper_name",
            ),
        )
        best_by_model = []
        promoted_types = [
            row["model_type"]
            for row in model_report["families"][family]["promoted_models"]
        ]
        for model_type in promoted_types:
            candidates = [row for row in ranking if row["model_type"] == model_type]
            best = candidates[0]
            best["hyperparameters"] = next(
                item
                for item in hyperparameter_grid(model_type)
                if item["name"] == best["hyper_name"]
            )
            best_by_model.append(best)
        best_by_model.sort(key=lambda row: -row["fold_mean"])
        finalists = best_by_model[: config.finalists_per_family]
        report["families"][family] = {
            "ranking": ranking,
            "best_by_model": best_by_model,
            "finalists": finalists,
            "halving_history": halving_history,
            "trial_count": len(records),
            "trial_scores": [
                {
                    "model_type": row["model_type"],
                    "feature_key": row["feature_key"],
                    "hyper_name": row["hyper_name"],
                    "score_fold": row["score_fold"],
                    "tsroc": row["tsroc"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }
                for row in records
            ],
        }
    _atomic_json(config.artifact_path / "tuning/summary.json", report)
    return report


def _finalist_id(row: dict[str, Any]) -> str:
    return f"{row['model_type']}__{row['feature_key']}__{row['hyper_name']}"


def run_finalist_cross_validation(context: Model6Context) -> dict[str, Any]:
    """Create complete five-fold OOF predictions and saved deployment voters."""

    tuning_path = context.config.artifact_path / "tuning/summary.json"
    if not tuning_path.exists():
        run_hyperparameter_tuning(context)
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    config = context.config
    report: dict[str, Any] = {
        "stage": "finalist_five_fold_cv",
        "selection_metric": "TSROC",
        "estimate_note": (
            "Every fold prediction is leakage-free, but candidates were promoted "
            "using development data. The sealed outer holdout is the unbiased final estimate."
        ),
        "outer_holdout_read": False,
        "families": {},
    }
    development_rows = context.split.development_rows
    local_fold = context.split.row_inner_fold[development_rows]
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    for family in FAMILIES:
        components = []
        for finalist in tuning["families"][family]["finalists"]:
            component_id = _finalist_id(finalist)
            feature_contract = {
                "method": finalist["feature_method"],
                "count": finalist["feature_count_label"],
                "key": finalist["feature_key"],
            }
            oof = np.full(development_rows.size, np.nan, dtype=np.float32)
            fold_records = []
            for fold in range(5):
                result_path = (
                    config.artifact_path
                    / "final_cv"
                    / family
                    / component_id
                    / f"fold_{fold}.json"
                )
                record = _fit_and_score_feature_candidate(
                    context,
                    family=family,
                    fold=fold,
                    feature_contract=feature_contract,
                    model_type=finalist["model_type"],
                    hyperparameters=finalist["hyperparameters"],
                    train_cap=config.final_train_rows,
                    stop_cap=config.final_stop_rows,
                    rounds=config.final_rounds,
                    patience=config.final_patience,
                    result_path=result_path,
                    save_model=True,
                )
                prediction = np.load(
                    result_path.with_suffix(".prediction.f32.npy"),
                    allow_pickle=False,
                )
                oof[local_fold == fold] = prediction
                fold_records.append(record)
            if not np.all(np.isfinite(oof)):
                raise AssertionError(f"incomplete final OOF: {family}/{component_id}")
            directory = config.artifact_path / "final_cv" / family / component_id
            np.save(directory / "raw_oof.f32.npy", oof)
            nested_prediction, postprocess = _postprocess_oof(context, oof)
            np.save(directory / "nested_oof.f32.npy", nested_prediction)
            component = {
                "family": family,
                "component": component_id,
                "model_type": finalist["model_type"],
                "feature_contract": feature_contract,
                "hyperparameters": finalist["hyperparameters"],
                "raw_tsroc": {
                    **_mean_se([record["tsroc"] for record in fold_records]),
                    "pooled_development_oof": float(tsauc(target, oof, online_time)),
                },
                "postprocess": postprocess,
                "fold_records": fold_records,
                "ensemble_prediction_path": str(
                    (directory / "nested_oof.f32.npy").relative_to(config.project_root)
                ),
            }
            _atomic_json(directory / "summary.json", component)
            components.append(component)
            del oof, nested_prediction
            gc.collect()
        report["families"][family] = {
            "components": sorted(
                components,
                key=lambda row: -row["postprocess"]["nested_tsroc"]["fold_mean"],
            )
        }
    _atomic_json(config.artifact_path / "final_cv/summary.json", report)
    return report


def _ensemble_tuning_mask(
    context: Model6Context, *, series_per_fold: int = 240
) -> np.ndarray:
    """Select complete series, label-independently, for fast weight search."""

    series_fold = context.split.inner_series_fold
    selected_series = np.zeros(series_fold.size, dtype=bool)
    for fold in range(5):
        positions = np.flatnonzero(series_fold == fold)
        # Complete-series sampling preserves TSROC's cross-sectional strata
        # while making thousands of blend comparisons practical.
        chosen = _cap_rows(
            positions, series_per_fold, context.config.seed + 88_000 + fold
        )
        selected_series[chosen] = True
    development_rows = context.split.development_rows
    return selected_series[
        context.data.metadata["series_position"][development_rows]
    ]


def _apply_recipe(margins: dict[str, np.ndarray], recipe: dict[str, Any]) -> np.ndarray:
    current = margins[recipe["start_component"]].copy()
    for step in recipe["steps"]:
        alpha = float(step["new_component_weight"])
        current = (1.0 - alpha) * current + alpha * margins[step["component"]]
    return _logistic(current)


def _greedy_recipe(
    target: np.ndarray,
    online_time: np.ndarray,
    margins: dict[str, np.ndarray],
    tune_mask: np.ndarray,
    config: Model6Config,
) -> dict[str, Any]:
    """Greedily add diverse logit components using only selected tune rows."""

    names = sorted(margins)
    if not names:
        raise ValueError("ensemble needs at least one component")

    def score(values: np.ndarray) -> float:
        return float(tsauc(target[tune_mask], values[tune_mask], online_time[tune_mask]))

    individual = {name: score(_logistic(margins[name])) for name in names}
    start = max(names, key=lambda name: (individual[name], name))
    current_margin = margins[start].copy()
    current_score = individual[start]
    used = {start}
    steps = []
    grid = np.unique(
        np.r_[np.arange(config.ensemble_weight_step, 1.0, config.ensemble_weight_step), 1.0]
    )
    while len(used) < min(config.ensemble_max_components, len(names)):
        best: tuple[float, str, float, np.ndarray] | None = None
        for name in names:
            if name in used:
                continue
            for alpha in grid:
                candidate_margin = (1.0 - alpha) * current_margin + alpha * margins[name]
                candidate_score = score(_logistic(candidate_margin))
                row = (candidate_score, name, float(alpha), candidate_margin)
                if best is None or row[:3] > best[:3]:
                    best = row
        if best is None or best[0] <= current_score + 1e-5:
            break
        current_score, name, alpha, current_margin = best
        used.add(name)
        steps.append(
            {
                "component": name,
                "new_component_weight": alpha,
                "tuning_tsroc_after_step": current_score,
            }
        )
    return {
        "start_component": start,
        "individual_tuning_tsroc": individual,
        "steps": steps,
        "final_tuning_tsroc": current_score,
    }


def _nested_greedy_ensemble(
    context: Model6Context,
    component_paths: dict[str, str],
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    development_rows = context.split.development_rows
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    fold = context.split.row_inner_fold[development_rows]
    tuning_rows = _ensemble_tuning_mask(context)
    probabilities = {
        name: np.load(context.config.project_root / path, allow_pickle=False).astype(np.float64)
        for name, path in component_paths.items()
    }
    margins = {name: _safe_logit(value) for name, value in probabilities.items()}
    nested = np.empty(target.size, dtype=np.float32)
    held_recipes = []
    held_scores = []
    for held in range(5):
        recipe = _greedy_recipe(
            target,
            online_time,
            margins,
            tuning_rows & (fold != held),
            context.config,
        )
        prediction = _apply_recipe(margins, recipe)
        mask = fold == held
        nested[mask] = prediction[mask]
        held_score = float(tsauc(target[mask], prediction[mask], online_time[mask]))
        held_recipes.append({"held_fold": held, "recipe": recipe, "held_tsroc": held_score})
        held_scores.append(held_score)
    deployment_recipe = _greedy_recipe(
        target, online_time, margins, tuning_rows, context.config
    )
    postprocessed, postprocess = _postprocess_oof(context, nested)
    report = {
        "label": label,
        "method": "nested greedy logit ensemble",
        "weight_tuning_series_per_fold": 240,
        "components": component_paths,
        "held_recipes": held_recipes,
        "nested_raw_tsroc": {
            **_mean_se(held_scores),
            "pooled_development_oof": float(tsauc(target, nested, online_time)),
        },
        "postprocess": postprocess,
        "deployment_recipe": deployment_recipe,
    }
    return postprocessed, report


def run_ensembling(context: Model6Context) -> dict[str, Any]:
    """Build larger core/candidate ensembles, then a cross-family ensemble."""

    stack_path = context.config.artifact_path / "detector_stack/summary.json"
    final_path = context.config.artifact_path / "final_cv/summary.json"
    if not stack_path.exists():
        run_detector_stack(context)
    if not final_path.exists():
        run_finalist_cross_validation(context)
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "stage": "nested_ensembling",
        "target_tsroc": context.config.target_tsroc,
        "outer_holdout_read": False,
        "families": {},
    }
    family_prediction_paths: dict[str, str] = {}
    global_paths: dict[str, str] = {}
    for family in FAMILIES:
        paths = {}
        for component in stack["families"][family]["components"]:
            name = f"stack__{component['component']}"
            paths[name] = component["ensemble_prediction_path"]
        for component in final["families"][family]["components"]:
            name = f"feature__{component['component']}"
            paths[name] = component["ensemble_prediction_path"]
        prediction, family_report = _nested_greedy_ensemble(
            context, paths, f"{family}_larger_model"
        )
        directory = context.config.artifact_path / "ensemble" / family
        directory.mkdir(parents=True, exist_ok=True)
        prediction_path = directory / "nested_oof.f32.npy"
        np.save(prediction_path, prediction)
        family_report["ensemble_prediction_path"] = str(
            prediction_path.relative_to(context.config.project_root)
        )
        _atomic_json(directory / "summary.json", family_report)
        report["families"][family] = family_report
        family_prediction_paths[family] = family_report["ensemble_prediction_path"]
        global_paths[f"family__{family}"] = family_report["ensemble_prediction_path"]
        # Global selection can also retain a strong individual that a family
        # greedy path omitted.
        global_paths.update({f"{family}__{key}": value for key, value in paths.items()})
    global_prediction, global_report = _nested_greedy_ensemble(
        context, global_paths, "core_candidate_global_ensemble"
    )
    global_directory = context.config.artifact_path / "ensemble" / "global"
    global_directory.mkdir(parents=True, exist_ok=True)
    global_path = global_directory / "nested_oof.f32.npy"
    np.save(global_path, global_prediction)
    global_report["ensemble_prediction_path"] = str(
        global_path.relative_to(context.config.project_root)
    )
    achieved = float(global_report["postprocess"]["nested_tsroc"]["fold_mean"])
    global_report["target_tsroc"] = context.config.target_tsroc
    global_report["target_reached_on_nested_development_cv"] = bool(
        achieved >= context.config.target_tsroc
    )
    _atomic_json(global_directory / "summary.json", global_report)
    report["global"] = global_report
    _atomic_json(context.config.artifact_path / "ensemble/summary.json", report)
    return report


def freeze_deployment(context: Model6Context) -> dict[str, Any]:
    """Hash all deployment choices before any outer-holdout target is indexed."""

    ensemble_path = context.config.artifact_path / "ensemble/summary.json"
    if not ensemble_path.exists():
        run_ensembling(context)
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    final = json.loads(
        (context.config.artifact_path / "final_cv/summary.json").read_text(
            encoding="utf-8"
        )
    )
    stack = json.loads(
        (context.config.artifact_path / "detector_stack/summary.json").read_text(
            encoding="utf-8"
        )
    )
    contract = {
        "experiment_version": MODEL6_VERSION,
        "created_without_outer_holdout_scoring": True,
        "source_selection_hash": context.source_selection["selection_hash"],
        "target_tsroc": context.config.target_tsroc,
        "development_nested_tsroc": ensemble["global"]["postprocess"][
            "nested_tsroc"
        ],
        "detector_stack": stack["families"],
        "feature_finalists": final["families"],
        "family_ensembles": ensemble["families"],
        "global_ensemble": ensemble["global"],
        "holdout_inference": (
            "Average five saved score-fold-excluded voters per feature model and "
            "per detector bank, apply frozen stack/greedy recipes and causal "
            "transforms, then score exactly once."
        ),
        "bootstrap_replicates": 200,
    }
    contract["contract_hash"] = _hash_json(contract)
    path = context.config.artifact_path / "frozen_contract.json"
    _write_or_validate(path, contract)
    return contract


def _deployment_transform(
    prediction: np.ndarray,
    series_position: np.ndarray,
    component_report: dict[str, Any],
) -> np.ndarray:
    postprocess = component_report["postprocess"]
    specification = postprocess.get("deployment_transform_spec")
    if specification is None:
        # Read old Model 6 reports without silently changing their semantics.
        penalty = postprocess["deployment_penalty"]
        return _logistic(
            causal_decayed_max(prediction, series_position, penalty)
        ).astype(np.float32)

    kind = specification["kind"]
    if kind == "raw":
        transformed = np.asarray(prediction, dtype=np.float64)
    elif kind == "logit_subtractive_decayed_max":
        transformed = _logistic(
            causal_decayed_max(
                prediction,
                series_position,
                float(specification["penalty"]),
            )
        )
    elif kind == "probability_multiplicative_decayed_max":
        transformed = causal_probability_decayed_max(
            prediction,
            series_position,
            float(specification["decay_factor"]),
        )
    else:
        raise ValueError(f"unknown postprocess transform kind: {kind}")
    return np.asarray(transformed, dtype=np.float32)


def _predict_detector_banks_on_holdout(
    context: Model6Context, family: str
) -> tuple[np.ndarray, list[str]]:
    """Bag five Model 5 voters for every selected bank on the sealed inputs."""

    banks = list(context.family_banks[family])
    holdout_rows = context.split.holdout_rows
    output = np.empty((holdout_rows.size, len(banks)), dtype=np.float32)
    cache_root = context.config.artifact_path / "holdout_predictions" / "detectors" / family
    cache_root.mkdir(parents=True, exist_ok=True)
    for bank_index, bank in enumerate(banks):
        cache_path = cache_root / f"{bank}.f32.npy"
        if cache_path.exists():
            prediction = np.load(cache_path, allow_pickle=False)
            if prediction.shape != (holdout_rows.size,):
                raise ValueError(f"bad cached detector holdout prediction: {cache_path}")
            output[:, bank_index] = prediction
            print(f"holdout detector {family}/{bank}: checkpoint", flush=True)
            continue
        source_directory = context.source_config.artifact_path / "screen" / family / bank
        manifest = json.loads(
            (source_directory / "manifest.json").read_text(encoding="utf-8")
        )
        columns = np.asarray(manifest["columns"], dtype=np.int32)
        fold_margins = np.empty((holdout_rows.size, 5), dtype=np.float32)
        for fold in range(5):
            booster = lgb.Booster(model_file=str(source_directory / f"fold_{fold}.txt"))
            prediction = base.predict_rows(
                booster,
                context.data.joint_matrix,
                holdout_rows,
                columns,
                context.config.prediction_chunk_rows,
            )
            fold_margins[:, fold] = _safe_logit(prediction).astype(np.float32)
            del booster, prediction
            gc.collect()
        prediction = _logistic(fold_margins.mean(axis=1)).astype(np.float32)
        np.save(cache_path, prediction)
        output[:, bank_index] = prediction
        print(f"holdout detector {family}/{bank}: complete", flush=True)
        del fold_margins, prediction
        gc.collect()
    return output, banks


def _deployment_detector_meta_prediction(
    context: Model6Context,
    family: str,
    model_type: str,
    development_stack: np.ndarray,
    holdout_stack: np.ndarray,
) -> np.ndarray:
    """Fit a frozen second level on development OOF and predict holdout inputs."""

    directory = (
        context.config.artifact_path
        / "holdout_predictions"
        / "detector_meta"
        / family
        / model_type
    )
    prediction_path = directory / "holdout.f32.npy"
    model_path = directory / "deployment.joblib"
    result_path = directory / "manifest.json"
    development_rows = context.split.development_rows
    local_fold = context.split.row_inner_fold[development_rows]
    target = context.data.metadata["target"][development_rows]
    online_time = context.data.metadata["time_online"][development_rows]
    stop_fold = 4
    stop_mask = local_fold == stop_fold
    train_mask = ~stop_mask
    train_positions = _cap_rows(
        np.flatnonzero(train_mask),
        context.config.detector_stack_train_rows,
        context.config.seed + 93_001,
    )
    stop_positions = _cap_rows(
        np.flatnonzero(stop_mask),
        context.config.detector_stack_train_rows // 2,
        context.config.seed + 93_002,
    )
    hyper = _base_hyperparameters(model_type)
    contract = {
        "stage": "deployment_detector_meta",
        "family": family,
        "model_type": model_type,
        "banks": list(context.family_banks[family]),
        "hyperparameters": hyper,
        "stop_fold": stop_fold,
        "train_rows": int(train_positions.size),
        "stop_rows": int(stop_positions.size),
        "source": "development OOF detector predictions only",
        "holdout_labels_read": False,
    }
    contract_hash = _hash_json(contract)
    if prediction_path.exists() and model_path.exists() and result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("contract_hash") != contract_hash:
            raise ValueError(f"stale detector meta deployment: {result_path}")
        return np.load(prediction_path, allow_pickle=False).astype(np.float32)
    train_weight = pair_weights_from_reference(
        target[train_positions],
        online_time[train_positions],
        target[train_mask],
        online_time[train_mask],
    )
    stop_weight = pair_weights_from_reference(
        target[stop_positions],
        online_time[stop_positions],
        target[stop_mask],
        online_time[stop_mask],
    )
    fitted = _fit_model(
        model_type=model_type,
        hyperparameters=hyper,
        x_train=development_stack[train_positions],
        y_train=target[train_positions],
        train_weight=train_weight,
        x_stop=development_stack[stop_positions],
        y_stop=target[stop_positions],
        stop_weight=stop_weight,
        stop_time=online_time[stop_positions],
        rounds=context.config.model_search_rounds,
        patience=context.config.model_search_patience,
        seed=context.config.seed + 93_003,
        threads=context.config.num_threads,
        device=context.actual_device,
    )
    prediction = _predict_array(fitted, holdout_stack).astype(np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, model_path, compress=0)
    np.save(prediction_path, prediction)
    _atomic_json(result_path, {**contract, "contract_hash": contract_hash})
    return prediction


def _holdout_stack_components(
    context: Model6Context,
    family: str,
    stack_report: dict[str, Any],
) -> dict[str, np.ndarray]:
    holdout_probability, banks = _predict_detector_banks_on_holdout(context, family)
    development_probability, development_banks, _ = _detector_oof_matrix(context, family)
    if banks != development_banks:
        raise ValueError("development and holdout detector bank orders differ")
    holdout_owner = context.data.metadata["series_position"][context.split.holdout_rows]
    by_name = {row["component"]: row for row in stack_report["components"]}
    output: dict[str, np.ndarray] = {}
    for name, component in by_name.items():
        if name == "logit_mean":
            raw = _logistic(_safe_logit(holdout_probability).mean(axis=1))
        elif name == "nested_top_n":
            order = [banks.index(bank) for bank in component["deployment_bank_order"]]
            count = int(component["deployment_count"])
            raw = _logistic(
                _safe_logit(holdout_probability[:, order[:count]]).mean(axis=1)
            )
        elif name == "greedy_banks":
            bank_margins = {
                bank: _safe_logit(holdout_probability[:, index])
                for index, bank in enumerate(banks)
            }
            raw = _apply_recipe(bank_margins, component["deployment_recipe"])
        elif name.startswith("meta_"):
            model_type = name.removeprefix("meta_")
            raw = _deployment_detector_meta_prediction(
                context,
                family,
                model_type,
                _stack_features(development_probability),
                _stack_features(holdout_probability),
            )
        else:
            raise ValueError(f"unknown detector stack component: {name}")
        output[f"stack__{name}"] = _deployment_transform(
            np.asarray(raw), holdout_owner, component
        )
    return output


def _holdout_feature_components(
    context: Model6Context,
    family: str,
    final_report: dict[str, Any],
) -> dict[str, np.ndarray]:
    holdout_rows = context.split.holdout_rows
    holdout_owner = context.data.metadata["series_position"][holdout_rows]
    output: dict[str, np.ndarray] = {}
    for component in final_report["components"]:
        component_id = component["component"]
        cache_path = (
            context.config.artifact_path
            / "holdout_predictions"
            / "feature_models"
            / family
            / f"{component_id}.f32.npy"
        )
        if cache_path.exists():
            raw = np.load(cache_path, allow_pickle=False)
        else:
            directory = context.config.artifact_path / "final_cv" / family / component_id
            fold_margins = np.empty((holdout_rows.size, 5), dtype=np.float32)
            for fold in range(5):
                fitted: FittedModel = joblib.load(directory / f"fold_{fold}.joblib")
                with np.load(
                    directory / f"fold_{fold}.columns.npz", allow_pickle=False
                ) as saved:
                    columns = saved["columns"].astype(np.int32)
                prediction = _predict_feature_rows(
                    context, fitted, holdout_rows, columns
                )
                fold_margins[:, fold] = _safe_logit(prediction).astype(np.float32)
                del fitted, prediction
                gc.collect()
            raw = _logistic(fold_margins.mean(axis=1)).astype(np.float32)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, raw)
            del fold_margins
        output[f"feature__{component_id}"] = _deployment_transform(
            raw, holdout_owner, component
        )
    return output


def build_frozen_holdout_prediction(
    context: Model6Context, frozen: dict[str, Any]
) -> dict[str, Any]:
    """Generate the frozen holdout score vector without indexing its labels."""

    path = context.config.artifact_path / "frozen_contract.json"
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != frozen:
        raise ValueError("supplied frozen contract differs from disk")
    unhashed = dict(frozen)
    claimed = unhashed.pop("contract_hash")
    if claimed != _hash_json(unhashed):
        raise ValueError("frozen contract hash is invalid")
    prediction_manifest_path = (
        context.config.artifact_path / "holdout_predictions/manifest.json"
    )
    final_prediction_path = (
        context.config.artifact_path / "holdout_predictions/global.f32.npy"
    )
    if prediction_manifest_path.exists() and final_prediction_path.exists():
        manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("frozen_contract_hash") != frozen["contract_hash"]:
            raise ValueError("holdout prediction belongs to another frozen contract")
        return manifest

    holdout_owner = context.data.metadata["series_position"][context.split.holdout_rows]
    family_outputs: dict[str, np.ndarray] = {}
    all_components: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        components = _holdout_stack_components(
            context, family, frozen["detector_stack"][family]
        )
        components.update(
            _holdout_feature_components(
                context, family, frozen["feature_finalists"][family]
            )
        )
        margins = {name: _safe_logit(value) for name, value in components.items()}
        family_raw = _apply_recipe(
            margins, frozen["family_ensembles"][family]["deployment_recipe"]
        )
        family_prediction = _deployment_transform(
            family_raw, holdout_owner, frozen["family_ensembles"][family]
        )
        family_outputs[family] = family_prediction
        all_components[f"family__{family}"] = family_prediction
        all_components.update(
            {f"{family}__{name}": value for name, value in components.items()}
        )
    global_margins = {
        name: _safe_logit(value) for name, value in all_components.items()
    }
    global_raw = _apply_recipe(
        global_margins, frozen["global_ensemble"]["deployment_recipe"]
    )
    prediction = _deployment_transform(
        global_raw, holdout_owner, frozen["global_ensemble"]
    )
    final_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(final_prediction_path, prediction.astype(np.float32))
    manifest = {
        "experiment_version": MODEL6_VERSION,
        "frozen_contract_hash": frozen["contract_hash"],
        "prediction_path": str(final_prediction_path.relative_to(context.config.project_root)),
        "holdout_rows": int(context.split.holdout_rows.size),
        "holdout_labels_read": False,
        "five_fold_voters_bagged": True,
    }
    _atomic_json(prediction_manifest_path, manifest)
    return manifest


def evaluate_frozen_holdout(
    context: Model6Context,
    frozen: dict[str, Any],
    prediction_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Open the sealed outer target exactly once and report cluster-bootstrap SE."""

    result_path = context.config.artifact_path / "holdout_results.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("frozen_contract_hash") != frozen["contract_hash"]:
            raise ValueError("saved holdout result belongs to another frozen contract")
        return saved
    if prediction_manifest.get("frozen_contract_hash") != frozen["contract_hash"]:
        raise ValueError("prediction manifest and frozen contract differ")
    prediction = np.load(
        context.config.project_root / prediction_manifest["prediction_path"],
        allow_pickle=False,
    )
    rows = context.split.holdout_rows
    if prediction.shape != (rows.size,):
        raise ValueError("frozen holdout prediction has the wrong number of rows")
    # This is deliberately the only Model 6 function that indexes holdout labels.
    target = context.data.metadata["target"][rows]
    online_time = context.data.metadata["time_online"][rows]
    owner = context.data.metadata["series_position"][rows]
    result = {
        "experiment_version": MODEL6_VERSION,
        "frozen_contract_hash": frozen["contract_hash"],
        "holdout_series": int(context.split.holdout_series_mask.sum()),
        "holdout_rows": int(rows.size),
        "holdout_tsroc": float(tsauc(target, prediction, online_time)),
        "ordinary_roc_diagnostic": float(roc_auc_score(target, prediction)),
        "complete_series_bootstrap": base.cluster_bootstrap_tsauc(
            target,
            prediction,
            online_time,
            owner,
            replicates=200,
            seed=context.config.seed + 99_999,
        ),
        "target_tsroc": context.config.target_tsroc,
    }
    result["target_reached"] = bool(
        result["holdout_tsroc"] >= context.config.target_tsroc
    )
    _atomic_json(result_path, result)
    return result


def run_development(context: Model6Context) -> dict[str, Any]:
    """Run/resume every development-only stage in dependency order."""

    run_detector_stack(context)
    run_feature_search(context)
    run_model_search(context)
    run_hyperparameter_tuning(context)
    run_finalist_cross_validation(context)
    return run_ensembling(context)


def run_all(context: Model6Context) -> dict[str, Any]:
    """Run development, freeze once, build predictions, and open holdout once."""

    run_development(context)
    frozen = freeze_deployment(context)
    prediction_manifest = build_frozen_holdout_prediction(context, frozen)
    return evaluate_frozen_holdout(context, frozen, prediction_manifest)
