"""Leakage-controlled non-LightGBM model study for submission 4.

This experiment inherits submission 3's train-derived feature contracts and
series-level split without changing either one:

* folds 2, 3, 4: model fitting;
* fold 1: hyperparameter and ensemble selection;
* fold 0: locked comparison test, read only after ``freeze``.

The four requested learners are evaluated independently on the selected core
and candidate feature families: a standardized SGD logistic linear baseline,
XGBoost, CatBoost, and Random Forest.  Hyperparameter screening uses exactly
100,000 label-independent sampled fitting rows and 200,000 label-independent
validation rows.  The winning configuration for each learner/family is then
refit on the complete deployment-size row sample and scored on every fold-1
row.  Only those full-validation predictions are allowed to choose the final
core/candidate blend.

Run from the repository root with the isolated environment containing XGBoost
and CatBoost::

    /tmp/submission4-venv/bin/python alternative_model_experiment.py tune-family --family core
    /tmp/submission4-venv/bin/python alternative_model_experiment.py tune-family --family candidate
    /tmp/submission4-venv/bin/python alternative_model_experiment.py refit-family --family core
    /tmp/submission4-venv/bin/python alternative_model_experiment.py refit-family --family candidate
    /tmp/submission4-venv/bin/python alternative_model_experiment.py freeze
    /tmp/submission4-venv/bin/python alternative_model_experiment.py evaluate-test

Each command checkpoints JSON records so completed work is not repeated.  The
``freeze`` and ``evaluate-test`` commands intentionally refuse to overwrite
their outputs: changing a model after seeing the comparison test would turn
that test into another validation set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from feature_pipeline import feature_names, load_feature_cache
from train_cv import (
    deterministic_sample_mask,
    make_series_folds,
    matrix_rows,
    pair_weights_from_reference,
)
from tsauc import tsauc


EXPERIMENT_VERSION = "alternative-models-v1"
ARTIFACT_ROOT = Path("models/artifacts/alternative_models_v2")
FEATURE_SELECTION_ROOT = Path("models/artifacts/feature_selection_v1")

FOLD_SEED = 73051
ROW_SAMPLE_SEED = 99173
MODEL_SEED = 94117
TRAIN_FOLDS = (2, 3, 4)
VALID_FOLD = 1
TEST_FOLD = 0
DEFAULT_SAMPLE_FRACTION = 0.20
RANDOM_FOREST_SAMPLE_FRACTION = 0.08
TUNE_TRAIN_ROWS = 100_000
TUNE_VALID_ROWS = 200_000
PREDICTION_CHUNK_ROWS = 25_000
THREADS = max(1, min(8, os.cpu_count() or 1))

MODEL_TYPES = ("linear", "xgboost", "catboost", "random_forest")
FAMILY_SPEC = {
    "core": {
        "cache": "core_features",
        "selection": "consensus_top_768",
    },
    "candidate": {
        "cache": "candidate_features",
        "selection": "gain_corr_0p98",
    },
}


@dataclass
class FamilyData:
    """One selected feature family and its immutable series split."""

    family: str
    matrix: np.ndarray
    metadata: dict[str, np.ndarray]
    columns: np.ndarray
    names: list[str]
    row_fold: np.ndarray

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


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _column_hash(columns: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(columns, dtype=np.int32).tobytes()).hexdigest()


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(bounded) - np.log1p(-bounded)


def _logistic(margin: np.ndarray) -> np.ndarray:
    value = np.asarray(margin, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _sanitize(matrix: np.ndarray) -> np.ndarray:
    """Give every learner the same finite float32 feature matrix."""

    output = np.ascontiguousarray(matrix, dtype=np.float32)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return output


def _load_family(family: str) -> FamilyData:
    specification = FAMILY_SPEC[family]
    cache = load_feature_cache("models/artifacts", specification["cache"])
    metadata = cache.load_metadata()
    matrix = cache.open("r")
    selection_path = (
        FEATURE_SELECTION_ROOT
        / family
        / "selections"
        / f"{specification['selection']}.npz"
    )
    with np.load(selection_path, allow_pickle=False) as selection:
        columns = np.asarray(selection["columns"], dtype=np.int32)
        names = [str(name) for name in selection["names"]]
    available_names = feature_names(cache.banks)
    if names != [available_names[int(index)] for index in columns]:
        raise ValueError(f"{family} feature-selection contract is stale")

    series_fold = make_series_folds(
        metadata["online_lengths"],
        metadata["tau_index"],
        n_splits=5,
        seed=FOLD_SEED,
    )
    row_fold = series_fold[metadata["series_position"]]
    data = FamilyData(
        family=family,
        matrix=matrix,
        metadata=metadata,
        columns=columns,
        names=names,
        row_fold=row_fold,
    )
    _validate_inherited_contract(data)
    return data


def _validate_inherited_contract(data: FamilyData) -> None:
    inherited = json.loads((FEATURE_SELECTION_ROOT / "split.json").read_text())
    expected = {
        "train": int(data.train_full.sum()),
        "validation": int(data.valid_full.sum()),
        "test": int(data.test_full.sum()),
    }
    actual = {
        role: int(inherited["summary"][role]["rows"])
        for role in ("train", "validation", "test")
    }
    if expected != actual:
        raise ValueError("alternative experiment differs from the inherited split")
    if inherited["fold_seed"] != FOLD_SEED:
        raise ValueError("alternative experiment uses a different fold seed")
    if data.columns.size != len(data.names):
        raise ValueError("selected columns and feature names differ in length")


def _sample_fraction(model_type: str) -> float:
    if model_type == "random_forest":
        return RANDOM_FOREST_SAMPLE_FRACTION
    return DEFAULT_SAMPLE_FRACTION


def _role_rows(
    data: FamilyData,
    role_mask: np.ndarray,
    model_type: str,
) -> np.ndarray:
    sampled = deterministic_sample_mask(
        data.metadata["row_id"],
        data.metadata["time_online"],
        _sample_fraction(model_type),
        ROW_SAMPLE_SEED,
    )
    return np.flatnonzero(role_mask & sampled)


def _fixed_cap(rows: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Uniform exact-size subsample chosen without looking at target values."""

    rows = np.asarray(rows, dtype=np.int64)
    if rows.size <= maximum:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=maximum, replace=False))


def _weights(data: FamilyData, rows: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    return pair_weights_from_reference(
        data.metadata["target"][rows],
        data.metadata["time_online"][rows],
        data.metadata["target"][reference_mask],
        data.metadata["time_online"][reference_mask],
    )


def _configurations(model_type: str) -> list[dict[str, Any]]:
    """Small, predeclared grids; no fold-0 result can alter this search space."""

    if model_type == "linear":
        return [
            {
                "name": "linear_l2_a3e-6",
                "penalty": "l2",
                "alpha": 3e-6,
                "max_iter": 35,
                "average": True,
            },
            {
                "name": "linear_l2_a1e-5",
                "penalty": "l2",
                "alpha": 1e-5,
                "max_iter": 35,
                "average": True,
            },
            {
                "name": "linear_enet_a1e-5_l15",
                "penalty": "elasticnet",
                "alpha": 1e-5,
                "l1_ratio": 0.15,
                "max_iter": 35,
                "average": True,
            },
        ]
    if model_type == "xgboost":
        common = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_bin": 127,
            "subsample": 0.85,
            "n_jobs": THREADS,
        }
        return [
            {
                "name": "xgb_depth3_lr006",
                **common,
                "n_estimators": 180,
                "learning_rate": 0.06,
                "max_depth": 3,
                "min_child_weight": 10.0,
                "colsample_bytree": 0.70,
                "reg_alpha": 0.10,
                "reg_lambda": 5.0,
            },
            {
                "name": "xgb_depth5_lr004",
                **common,
                "n_estimators": 240,
                "learning_rate": 0.04,
                "max_depth": 5,
                "min_child_weight": 20.0,
                "colsample_bytree": 0.65,
                "reg_alpha": 0.20,
                "reg_lambda": 8.0,
            },
            {
                "name": "xgb_depth7_lr0035",
                **common,
                "n_estimators": 160,
                "learning_rate": 0.035,
                "max_depth": 7,
                "min_child_weight": 40.0,
                "colsample_bytree": 0.55,
                "reg_alpha": 0.20,
                "reg_lambda": 8.0,
            },
        ]
    if model_type == "catboost":
        common = {
            "loss_function": "Logloss",
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
            "thread_count": THREADS,
        }
        return [
            {
                "name": "cat_depth5_lr006",
                **common,
                "iterations": 200,
                "depth": 5,
                "learning_rate": 0.06,
                "l2_leaf_reg": 5.0,
                "random_strength": 0.5,
                "rsm": 0.70,
            },
            {
                "name": "cat_depth6_lr004",
                **common,
                "iterations": 260,
                "depth": 6,
                "learning_rate": 0.04,
                "l2_leaf_reg": 8.0,
                "random_strength": 0.5,
                "rsm": 0.65,
            },
            {
                "name": "cat_depth8_lr0035",
                **common,
                "iterations": 200,
                "depth": 8,
                "learning_rate": 0.035,
                "l2_leaf_reg": 10.0,
                "random_strength": 1.0,
                "rsm": 0.55,
            },
        ]
    if model_type == "random_forest":
        return [
            {
                "name": "rf64_depth14_leaf20_sqrt",
                "n_estimators": 64,
                "max_depth": 14,
                "min_samples_leaf": 20,
                "max_features": "sqrt",
                "max_samples": 0.75,
            },
            {
                "name": "rf64_full_leaf50_f005",
                "n_estimators": 64,
                "max_depth": None,
                "min_samples_leaf": 50,
                "max_features": 0.05,
                "max_samples": 0.80,
            },
        ]
    raise ValueError(f"unknown model type: {model_type}")


def _family_configurations(family: str, model_type: str) -> list[dict[str, Any]]:
    """Apply predeclared runtime caps for the wider candidate source cache."""

    configurations = _configurations(model_type)
    if family == "candidate" and model_type == "xgboost":
        configurations = [
            item for item in configurations if item["name"] != "xgb_depth7_lr0035"
        ]
    if family == "candidate" and model_type == "random_forest":
        configurations = [dict(item) for item in configurations]
        for item in configurations:
            item["n_estimators"] = 32
            item["name"] = item["name"].replace("rf64_", "rf32_")
    return configurations


def _grid_hash(family: str) -> str:
    return _sha256_json(
        {
            model_type: _family_configurations(family, model_type)
            for model_type in MODEL_TYPES
        }
    )


def _validate_tuning_report(family: str, report: dict[str, Any]) -> None:
    """Reject stale/extra configurations before they can affect selection."""

    if report.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError(f"incompatible tuning report for {family}")
    if report.get("grid_hash") != _grid_hash(family):
        raise ValueError(f"{family} tuning report has a different grid signature")
    for model_type, model_report in report.get("models", {}).items():
        allowed = {
            item["name"]: item
            for item in _family_configurations(family, model_type)
        }
        extra = set(model_report.get("configurations", {})) - set(allowed)
        if extra:
            raise ValueError(
                f"{family}/{model_type} has stale configurations: {sorted(extra)}"
            )
        selected = model_report.get("selected_configuration")
        if selected is not None:
            name = selected.get("name")
            if name not in allowed or selected != allowed[name]:
                raise ValueError(
                    f"{family}/{model_type} selected configuration is not allowed"
                )


def _fit_model(
    model_type: str,
    configuration: dict[str, Any],
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    parameters = {key: value for key, value in configuration.items() if key != "name"}
    if model_type == "linear":
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(copy=True)
        scaler.fit(matrix, sample_weight=weights)
        transformed = scaler.transform(matrix).astype(np.float32, copy=False)
        np.clip(transformed, -25.0, 25.0, out=transformed)
        model = SGDClassifier(
            loss="log_loss",
            tol=1e-4,
            random_state=MODEL_SEED,
            learning_rate="optimal",
            **parameters,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(transformed, target, sample_weight=weights)
        del transformed
        return {"model": model, "scaler": scaler}

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            eval_metric="logloss",
            random_state=MODEL_SEED,
            verbosity=0,
            **parameters,
        )
        model.fit(matrix, target, sample_weight=weights, verbose=False)
        return {"model": model, "scaler": None}

    if model_type == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            random_seed=MODEL_SEED,
            allow_writing_files=False,
            verbose=False,
            **parameters,
        )
        model.fit(matrix, target, sample_weight=weights, verbose=False)
        return {"model": model, "scaler": None}

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_jobs=THREADS,
            random_state=MODEL_SEED,
            criterion="gini",
            **parameters,
        )
        model.fit(matrix, target, sample_weight=weights)
        return {"model": model, "scaler": None}

    raise ValueError(f"unknown model type: {model_type}")


def _predict_array(model_type: str, fitted: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    if model_type == "linear":
        transformed = fitted["scaler"].transform(matrix).astype(np.float32, copy=False)
        np.clip(transformed, -25.0, 25.0, out=transformed)
        margin = fitted["model"].decision_function(transformed)
        prediction = _logistic(margin)
        del transformed
    else:
        prediction = fitted["model"].predict_proba(matrix)[:, 1]
    return np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1 - 1e-7)


def _predict_rows(
    data: FamilyData,
    rows: np.ndarray,
    model_type: str,
    fitted: dict[str, Any],
) -> np.ndarray:
    prediction = np.empty(rows.size, dtype=np.float32)
    for left in range(0, rows.size, PREDICTION_CHUNK_ROWS):
        right = min(left + PREDICTION_CHUNK_ROWS, rows.size)
        block = _sanitize(matrix_rows(data.matrix, rows[left:right], data.columns))
        prediction[left:right] = _predict_array(model_type, fitted, block).astype(
            np.float32
        )
        del block
    return prediction


def _subfold_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    online_time: np.ndarray,
    series_position: np.ndarray,
    n_groups: int = 5,
) -> dict[str, object]:
    unique = np.unique(series_position)
    group_by_series = {
        int(series): index % n_groups for index, series in enumerate(unique)
    }
    group = np.asarray(
        [group_by_series[int(series)] for series in series_position], dtype=np.int8
    )
    scores = np.asarray(
        [
            tsauc(
                target[group == index],
                prediction[group == index],
                online_time[group == index],
            )
            for index in range(n_groups)
        ],
        dtype=np.float64,
    )
    return {
        "series_subfold_scores": scores.tolist(),
        "series_subfold_mean": float(scores.mean()),
        "series_subfold_standard_deviation": float(scores.std(ddof=1)),
        "series_subfold_standard_error": float(
            scores.std(ddof=1) / math.sqrt(scores.size)
        ),
        "standard_error_note": (
            "SE of five deterministic, disjoint series-cluster subfold scores; "
            "not a row-level independence estimate."
        ),
    }


def _tuning_path(family: str) -> Path:
    return ARTIFACT_ROOT / family / "tuning_results.json"


def _refit_path(family: str) -> Path:
    return ARTIFACT_ROOT / family / "validation_results.json"


def tune_family(family: str) -> None:
    """Screen every requested learner/configuration without touching fold 0."""

    data = _load_family(family)
    report_path = _tuning_path(family)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        _validate_tuning_report(family, report)
    else:
        report = {
            "experiment_version": EXPERIMENT_VERSION,
            "family": family,
            "feature_selection_method": FAMILY_SPEC[family]["selection"],
            "feature_count": int(data.columns.size),
            "column_hash": _column_hash(data.columns),
            "train_role": "folds 2,3,4 only",
            "validation_role": "fold 1 only",
            "test_fold_read": False,
            "grid_hash": _grid_hash(family),
            "tune_train_cap": TUNE_TRAIN_ROWS,
            "tune_validation_cap": TUNE_VALID_ROWS,
            "models": {},
        }

    valid_rows = _fixed_cap(
        np.flatnonzero(data.valid_full), TUNE_VALID_ROWS, MODEL_SEED + 301
    )
    x_valid = _sanitize(matrix_rows(data.matrix, valid_rows, data.columns))
    y_valid = data.metadata["target"][valid_rows]
    t_valid = data.metadata["time_online"][valid_rows]

    for model_index, model_type in enumerate(MODEL_TYPES):
        model_report = report["models"].setdefault(
            model_type,
            {
                "sample_fraction": _sample_fraction(model_type),
                "configurations": {},
            },
        )
        train_rows = _fixed_cap(
            _role_rows(data, data.train_full, model_type),
            TUNE_TRAIN_ROWS,
            MODEL_SEED + 1000 + model_index,
        )
        x_train = _sanitize(matrix_rows(data.matrix, train_rows, data.columns))
        y_train = data.metadata["target"][train_rows]
        weights = _weights(data, train_rows, data.train_full)
        print(
            f"{family}/{model_type}: {train_rows.size:,} tuning rows x "
            f"{data.columns.size:,} features; {valid_rows.size:,} validation rows",
            flush=True,
        )
        for configuration in _family_configurations(family, model_type):
            name = configuration["name"]
            if name in model_report["configurations"]:
                print(f"{family}/{name}: checkpoint exists", flush=True)
                continue
            started = time.perf_counter()
            fitted = _fit_model(
                model_type, configuration, x_train, y_train, weights
            )
            prediction = _predict_array(model_type, fitted, x_valid)
            score = tsauc(y_valid, prediction, t_valid)
            elapsed = time.perf_counter() - started
            model_report["configurations"][name] = {
                "configuration": configuration,
                "tune_validation_tsauc": float(score),
                "train_rows": int(train_rows.size),
                "validation_rows": int(valid_rows.size),
                "elapsed_seconds": float(elapsed),
            }
            _json_dump(report_path, report)
            print(
                f"{family}/{name}: tune TSAUC={score:.6f}, "
                f"elapsed={elapsed / 60:.2f} min",
                flush=True,
            )
            del fitted, prediction
            gc.collect()
        allowed_names = {
            item["name"]
            for item in _family_configurations(family, model_type)
        }
        ranked = sorted(
            (
                row
                for name, row in model_report["configurations"].items()
                if name in allowed_names
            ),
            key=lambda row: -row["tune_validation_tsauc"],
        )
        model_report["selected_configuration"] = ranked[0]["configuration"]
        model_report["selected_tune_validation_tsauc"] = ranked[0][
            "tune_validation_tsauc"
        ]
        _json_dump(report_path, report)
        del x_train, y_train, weights
        gc.collect()
    del x_valid
    print(json.dumps(report, indent=2), flush=True)


def refit_family(family: str) -> None:
    """Refit each tuned learner and score all fold-1 validation rows."""

    data = _load_family(family)
    tuning_path = _tuning_path(family)
    if not tuning_path.exists():
        raise RuntimeError(f"run tune-family first: {tuning_path}")
    tuning = json.loads(tuning_path.read_text())
    _validate_tuning_report(family, tuning)
    report_path = _refit_path(family)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("experiment_version") != EXPERIMENT_VERSION:
            raise ValueError(f"incompatible checkpoint: {report_path}")
    else:
        report = {
            "experiment_version": EXPERIMENT_VERSION,
            "family": family,
            "feature_selection_method": FAMILY_SPEC[family]["selection"],
            "feature_count": int(data.columns.size),
            "column_hash": _column_hash(data.columns),
            "hyperparameters_selected_on": "label-independent subset of fold 1",
            "full_validation_role": "every row in fold 1",
            "test_fold_read": False,
            "models": {},
        }

    valid_rows = np.flatnonzero(data.valid_full)
    y_valid = data.metadata["target"][valid_rows]
    t_valid = data.metadata["time_online"][valid_rows]
    s_valid = data.metadata["series_position"][valid_rows]

    for model_type in MODEL_TYPES:
        if model_type in report["models"]:
            print(f"{family}/{model_type}: full-validation checkpoint exists")
            continue
        configuration = tuning["models"][model_type]["selected_configuration"]
        train_rows = _role_rows(data, data.train_full, model_type)
        x_train = _sanitize(matrix_rows(data.matrix, train_rows, data.columns))
        y_train = data.metadata["target"][train_rows]
        weights = _weights(data, train_rows, data.train_full)
        print(
            f"{family}/{model_type}: refitting {configuration['name']} on "
            f"{train_rows.size:,} rows x {data.columns.size:,}",
            flush=True,
        )
        started = time.perf_counter()
        fitted = _fit_model(
            model_type, configuration, x_train, y_train, weights
        )
        fit_elapsed = time.perf_counter() - started
        model_directory = ARTIFACT_ROOT / family / "models"
        model_directory.mkdir(parents=True, exist_ok=True)
        model_path = model_directory / f"{model_type}.joblib"
        joblib.dump(fitted, model_path, compress=0)
        del x_train, y_train, weights
        gc.collect()

        started = time.perf_counter()
        prediction = _predict_rows(data, valid_rows, model_type, fitted)
        predict_elapsed = time.perf_counter() - started
        score = tsauc(y_valid, prediction, t_valid)
        prediction_path = (
            ARTIFACT_ROOT / family / "predictions" / f"{model_type}.validation.npy"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prediction_path, prediction.astype(np.float32))
        report["models"][model_type] = {
            "configuration": configuration,
            "sample_fraction": _sample_fraction(model_type),
            "training_rows": int(train_rows.size),
            "validation_rows": int(valid_rows.size),
            "validation_tsauc": float(score),
            **_subfold_summary(y_valid, prediction, t_valid, s_valid),
            "fit_seconds": float(fit_elapsed),
            "full_validation_predict_seconds": float(predict_elapsed),
            "model_path": str(model_path),
            "model_bytes": int(model_path.stat().st_size),
            "validation_prediction_path": str(prediction_path),
        }
        _json_dump(report_path, report)
        print(
            f"{family}/{model_type}: full validation TSAUC={score:.6f}, "
            f"SE={report['models'][model_type]['series_subfold_standard_error']:.6f}, "
            f"fit={fit_elapsed / 60:.2f} min, predict={predict_elapsed / 60:.2f} min",
            flush=True,
        )
        del fitted, prediction
        gc.collect()
    print(json.dumps(report, indent=2), flush=True)


def _load_validation_prediction(record: dict[str, Any]) -> np.ndarray:
    return np.load(record["validation_prediction_path"], allow_pickle=False)


def freeze_selection() -> None:
    """Choose and hash the deployment using fold 1; never index fold 0."""

    frozen_path = ARTIFACT_ROOT / "frozen_selection.json"
    if frozen_path.exists():
        raise FileExistsError(
            f"{frozen_path} already exists; do not revise a frozen choice"
        )
    reports = {family: json.loads(_refit_path(family).read_text()) for family in FAMILY_SPEC}
    for family, report in reports.items():
        missing = set(MODEL_TYPES) - set(report["models"])
        if missing:
            raise RuntimeError(f"{family} refits are incomplete: {sorted(missing)}")

    data = _load_family("core")
    valid_rows = np.flatnonzero(data.valid_full)
    target = data.metadata["target"][valid_rows]
    online_time = data.metadata["time_online"][valid_rows]
    series_position = data.metadata["series_position"][valid_rows]
    prediction = {
        (family, model_type): _load_validation_prediction(
            reports[family]["models"][model_type]
        )
        for family in FAMILY_SPEC
        for model_type in MODEL_TYPES
    }
    margins = {key: _safe_logit(value) for key, value in prediction.items()}
    weights = np.linspace(0.0, 1.0, 21)
    search: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for core_type in MODEL_TYPES:
        for candidate_type in MODEL_TYPES:
            for core_weight in weights:
                combined = _logistic(
                    core_weight * margins[("core", core_type)]
                    + (1.0 - core_weight) * margins[("candidate", candidate_type)]
                )
                score = tsauc(target, combined, online_time)
                record = {
                    "core_model_type": core_type,
                    "candidate_model_type": candidate_type,
                    "core_weight": float(core_weight),
                    "candidate_weight": float(1.0 - core_weight),
                    "validation_tsauc": float(score),
                }
                search.append(record)
                tie_key = (score, -abs(float(core_weight) - 0.5), -MODEL_TYPES.index(core_type))
                if best is None:
                    best = record
                else:
                    best_key = (
                        best["validation_tsauc"],
                        -abs(float(best["core_weight"]) - 0.5),
                        -MODEL_TYPES.index(str(best["core_model_type"])),
                    )
                    if tie_key > best_key:
                        best = record
    assert best is not None
    selected_prediction = _logistic(
        float(best["core_weight"])
        * margins[("core", str(best["core_model_type"]))]
        + float(best["candidate_weight"])
        * margins[("candidate", str(best["candidate_model_type"]))]
    )
    selected = {
        **best,
        "core_configuration": reports["core"]["models"][
            str(best["core_model_type"])
        ]["configuration"],
        "candidate_configuration": reports["candidate"]["models"][
            str(best["candidate_model_type"])
        ]["configuration"],
        "core_feature_selection": FAMILY_SPEC["core"]["selection"],
        "candidate_feature_selection": FAMILY_SPEC["candidate"]["selection"],
        "core_feature_count": reports["core"]["feature_count"],
        "candidate_feature_count": reports["candidate"]["feature_count"],
        **_subfold_summary(
            target, selected_prediction, online_time, series_position
        ),
    }
    contract: dict[str, object] = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_without_alternative_test_labels": True,
        "inherited_feature_contract_hash": json.loads(
            (FEATURE_SELECTION_ROOT / "frozen_selection.json").read_text()
        )["contract_hash"],
        "training_split": "folds 2,3,4",
        "selection_split": "fold 1",
        "locked_test_split": "fold 0",
        "model_types": list(MODEL_TYPES),
        "ensemble_weight_grid": weights.tolist(),
        "selected": selected,
        "individual_validation": {
            family: {
                model_type: {
                    "validation_tsauc": reports[family]["models"][model_type][
                        "validation_tsauc"
                    ],
                    "series_subfold_standard_error": reports[family]["models"]
                    [model_type]["series_subfold_standard_error"],
                    "configuration": reports[family]["models"][model_type][
                        "configuration"
                    ],
                }
                for model_type in MODEL_TYPES
            }
            for family in FAMILY_SPEC
        },
        "contract_hash": "",
    }
    contract["contract_hash"] = _sha256_json(
        {key: value for key, value in contract.items() if key != "contract_hash"}
    )
    _json_dump(ARTIFACT_ROOT / "ensemble_validation_search.json", search)
    _json_dump(frozen_path, contract)
    print(json.dumps(contract, indent=2), flush=True)


def _fit_development_component(
    data: FamilyData,
    model_type: str,
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], int, float]:
    rows = _role_rows(data, data.development_full, model_type)
    matrix = _sanitize(matrix_rows(data.matrix, rows, data.columns))
    target = data.metadata["target"][rows]
    weights = _weights(data, rows, data.development_full)
    started = time.perf_counter()
    fitted = _fit_model(model_type, configuration, matrix, target, weights)
    elapsed = time.perf_counter() - started
    del matrix, target, weights
    gc.collect()
    return fitted, int(rows.size), float(elapsed)


def _paired_difference(
    target: np.ndarray,
    selected: np.ndarray,
    baseline: np.ndarray,
    online_time: np.ndarray,
    series_position: np.ndarray,
) -> dict[str, object]:
    selected_summary = _subfold_summary(
        target, selected, online_time, series_position
    )
    baseline_summary = _subfold_summary(
        target, baseline, online_time, series_position
    )
    difference = np.asarray(
        selected_summary["series_subfold_scores"], dtype=np.float64
    ) - np.asarray(baseline_summary["series_subfold_scores"], dtype=np.float64)
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / math.sqrt(difference.size))
    critical = 2.7764451051977987
    return {
        "difference_by_series_subfold": difference.tolist(),
        "mean_difference": mean,
        "standard_error": se,
        "95_percent_t_interval": [mean - critical * se, mean + critical * se],
    }


def evaluate_locked_test() -> None:
    """Refit the frozen choice on train+validation and open fold 0 once."""

    report_path = ARTIFACT_ROOT / "locked_test_results.json"
    if report_path.exists():
        raise FileExistsError(
            f"{report_path} already exists; the locked test cannot be rerun"
        )
    frozen_path = ARTIFACT_ROOT / "frozen_selection.json"
    if not frozen_path.exists():
        raise RuntimeError("freeze the alternative model choice first")
    frozen = json.loads(frozen_path.read_text())
    expected_hash = _sha256_json(
        {key: value for key, value in frozen.items() if key != "contract_hash"}
    )
    if frozen["contract_hash"] != expected_hash:
        raise ValueError("frozen selection contract has been modified")

    data = {family: _load_family(family) for family in FAMILY_SPEC}
    if not np.array_equal(data["core"].row_fold, data["candidate"].row_fold):
        raise ValueError("core and candidate row splits differ")
    selected = frozen["selected"]
    test_rows = np.flatnonzero(data["core"].test_full)
    components: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    component_results: dict[str, object] = {}
    for family in FAMILY_SPEC:
        weight = float(selected[f"{family}_weight"])
        if weight <= 0.0:
            continue
        model_type = str(selected[f"{family}_model_type"])
        configuration = selected[f"{family}_configuration"]
        print(
            f"locked test: refitting {family}/{model_type} on development folds",
            flush=True,
        )
        fitted, training_rows, fit_seconds = _fit_development_component(
            data[family], model_type, configuration
        )
        model_path = ARTIFACT_ROOT / "locked_test_models" / f"{family}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, model_path, compress=0)
        started = time.perf_counter()
        prediction = _predict_rows(data[family], test_rows, model_type, fitted)
        predict_seconds = time.perf_counter() - started
        predictions[family] = prediction
        components[family] = fitted
        component_results[family] = {
            "model_type": model_type,
            "configuration": configuration,
            "weight": weight,
            "training_rows": training_rows,
            "fit_seconds": fit_seconds,
            "test_predict_seconds": float(predict_seconds),
            "model_path": str(model_path),
            "model_bytes": int(model_path.stat().st_size),
        }
        del fitted
        gc.collect()

    # This is the only point in this experiment that indexes fold-0 targets.
    target = data["core"].metadata["target"][test_rows]
    online_time = data["core"].metadata["time_online"][test_rows]
    series_position = data["core"].metadata["series_position"][test_rows]
    combined_margin = np.zeros(test_rows.size, dtype=np.float64)
    total_weight = 0.0
    for family, prediction in predictions.items():
        weight = float(selected[f"{family}_weight"])
        combined_margin += weight * _safe_logit(prediction)
        total_weight += weight
        component_results[family]["test_tsauc"] = float(
            tsauc(target, prediction, online_time)
        )
        component_results[family].update(
            _subfold_summary(target, prediction, online_time, series_position)
        )
    if total_weight <= 0:
        raise ValueError("frozen ensemble has no positive-weight component")
    ensemble = _logistic(combined_margin / total_weight)
    ensemble_score = tsauc(target, ensemble, online_time)
    result: dict[str, object] = {
        "experiment_version": EXPERIMENT_VERSION,
        "frozen_contract_hash": frozen["contract_hash"],
        "test_opened_after_freeze": True,
        "test_fold": TEST_FOLD,
        "test_series": int(np.unique(series_position).size),
        "test_rows": int(test_rows.size),
        "components": component_results,
        "ensemble_tsauc": float(ensemble_score),
        **_subfold_summary(target, ensemble, online_time, series_position),
    }
    baseline_path = FEATURE_SELECTION_ROOT / "selected.locked_test_prediction.f32.npy"
    if baseline_path.exists():
        baseline = np.load(baseline_path, allow_pickle=False)
        result["submission_3_reference"] = {
            "tsauc": float(tsauc(target, baseline, online_time)),
            "alternative_minus_submission_3": float(
                ensemble_score - tsauc(target, baseline, online_time)
            ),
            "paired_series_subfold": _paired_difference(
                target,
                ensemble,
                baseline,
                online_time,
                series_position,
            ),
        }
    prediction_path = ARTIFACT_ROOT / "selected.locked_test_prediction.f32.npy"
    np.save(prediction_path, ensemble.astype(np.float32))
    result["prediction_path"] = str(prediction_path)
    _json_dump(report_path, result)
    print(json.dumps(result, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("tune-family", "refit-family"):
        child = subparsers.add_parser(command)
        child.add_argument("--family", choices=tuple(FAMILY_SPEC), required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("evaluate-test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "tune-family":
        tune_family(args.family)
    elif args.command == "refit-family":
        refit_family(args.family)
    elif args.command == "freeze":
        freeze_selection()
    elif args.command == "evaluate-test":
        evaluate_locked_test()
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
