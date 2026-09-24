"""Leakage-safe feature-selection study for the real-time TSAUC model.

This experiment uses only feature caches built from ``train/``.  Complete
series, never rows, are assigned to three disjoint roles using the existing
five-fold stratification:

* folds 2, 3, 4: feature ranking and model fitting (60% of series);
* fold 1: method selection and early stopping (20% of series);
* fold 0: locked final test (20% of series).

The workflow is intentionally split into commands so test labels cannot be
consulted while feature-selection choices are still changing::

    python3 feature_selection_experiment.py run-family --family core
    python3 feature_selection_experiment.py run-family --family candidate
    python3 feature_selection_experiment.py freeze
    python3 feature_selection_experiment.py evaluate-test

``run-family`` compares representative embedded, model-explanation, filter,
control, and group-selection methods.  ``freeze`` searches core/candidate
validation combinations and writes an immutable decision contract without
reading fold-0 targets.  ``evaluate-test`` trains on folds 1--4 with fixed
columns and boosting rounds, then opens fold 0 exactly once.  It refuses to
overwrite an existing test report.
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

from feature_pipeline import bank_slices, feature_names, load_feature_cache
from train_cv import (
    DEFAULT_MODEL_PARAMETERS,
    deterministic_sample_mask,
    make_series_folds,
    make_tsauc_evaluator,
    matrix_rows,
    pair_weights_from_reference,
    predict_rows,
)
from tsauc import tsauc


EXPERIMENT_VERSION = "feature-selection-v1"
ARTIFACT_ROOT = Path("models/artifacts/feature_selection_v1")
FOLD_SEED = 73051
ROW_SAMPLE_FRACTION = 0.20
ROW_SAMPLE_SEED = 99173
MODEL_SEED = 61413
TRAIN_FOLDS = (2, 3, 4)
VALID_FOLD = 1
TEST_FOLD = 0
SHAP_ROWS = 8_192
CORRELATION_ROWS = 4_096
PREDICTION_CHUNK_ROWS = 50_000

FAMILY_CACHE = {
    "core": "core_features",
    "candidate": "candidate_features",
}


@dataclass
class FamilyData:
    """One feature family plus its immutable series-level split masks."""

    family: str
    cache: object
    matrix: np.ndarray
    metadata: dict[str, np.ndarray]
    names: list[str]
    banks: tuple[str, ...]
    row_fold: np.ndarray
    sampled: np.ndarray

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
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _column_hash(columns: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(columns, dtype=np.int32).tobytes()
    ).hexdigest()


def _safe_name(name: str) -> str:
    return name.replace(".", "p").replace("/", "_")


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


def _load_family(family: str) -> FamilyData:
    cache_name = FAMILY_CACHE[family]
    cache = load_feature_cache("models/artifacts", cache_name)
    metadata = cache.load_metadata()
    matrix = cache.open("r")
    names = feature_names(cache.banks)
    folds = make_series_folds(
        metadata["online_lengths"],
        metadata["tau_index"],
        n_splits=5,
        seed=FOLD_SEED,
    )
    row_fold = folds[metadata["series_position"]]
    sampled = deterministic_sample_mask(
        metadata["row_id"],
        metadata["time_online"],
        ROW_SAMPLE_FRACTION,
        ROW_SAMPLE_SEED,
    )
    data = FamilyData(
        family=family,
        cache=cache,
        matrix=matrix,
        metadata=metadata,
        names=names,
        banks=tuple(cache.banks),
        row_fold=row_fold,
        sampled=sampled,
    )
    _validate_split(data)
    return data


def _validate_split(data: FamilyData) -> None:
    series_fold = make_series_folds(
        data.metadata["online_lengths"],
        data.metadata["tau_index"],
        n_splits=5,
        seed=FOLD_SEED,
    )
    roles = {
        "train": np.isin(series_fold, TRAIN_FOLDS),
        "validation": series_fold == VALID_FOLD,
        "test": series_fold == TEST_FOLD,
    }
    if any(np.any(roles[left] & roles[right])
           for left, right in (("train", "validation"),
                               ("train", "test"),
                               ("validation", "test"))):
        raise AssertionError("series-level split roles overlap")
    if not np.all(roles["train"] | roles["validation"] | roles["test"]):
        raise AssertionError("some series have no split role")
    summary = {
        role: {
            "series": int(mask.sum()),
            "broken_series": int(
                np.sum(data.metadata["tau_index"][mask] >= 0)
            ),
            "rows": int(np.sum(data.metadata["online_lengths"][mask])),
        }
        for role, mask in roles.items()
    }
    expected = ARTIFACT_ROOT / "split.json"
    contract = {
        "experiment_version": EXPERIMENT_VERSION,
        "fold_seed": FOLD_SEED,
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "locked_test_fold": TEST_FOLD,
        "row_sample_fraction": ROW_SAMPLE_FRACTION,
        "row_sample_seed": ROW_SAMPLE_SEED,
        "summary": summary,
        "leakage_rule": (
            "Feature ranking and method selection may use train/validation "
            "only. Fold-0 targets may be read only by evaluate-test after "
            "frozen_selection.json exists."
        ),
    }
    if expected.exists():
        prior = json.loads(expected.read_text())
        if prior != contract:
            raise ValueError("saved split contract differs from current data")
    else:
        _json_dump(expected, contract)


def _parameters(seed: int) -> dict[str, object]:
    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    parameters.update(
        {
            "num_threads": max(1, min(8, os.cpu_count() or 1)),
            "seed": seed,
            "feature_fraction_seed": seed + 100,
            "bagging_seed": seed + 200,
            "data_random_seed": seed + 300,
        }
    )
    return parameters


def _family_directory(family: str) -> Path:
    return ARTIFACT_ROOT / family


def _selection_path(family: str, method: str) -> Path:
    return _family_directory(family) / "selections" / f"{_safe_name(method)}.npz"


def _save_selection(
    family: str,
    method: str,
    columns: np.ndarray,
    names: Sequence[str],
) -> Path:
    path = _selection_path(family, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        columns=np.asarray(columns, dtype=np.int32),
        names=np.asarray(list(names), dtype=np.str_),
    )
    return path


def _load_selection(family: str, method: str) -> tuple[np.ndarray, list[str]]:
    payload = np.load(_selection_path(family, method), allow_pickle=False)
    return payload["columns"].astype(np.int32), payload["names"].tolist()


def _results_path(family: str) -> Path:
    return _family_directory(family) / "validation_results.json"


def _eligible_for_deployment(method: str) -> bool:
    """Keep negative controls in the report but out of model selection."""

    return not method.startswith("random_control_")


def _load_results(family: str) -> dict[str, object]:
    path = _results_path(family)
    if not path.exists():
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "family": family,
            "methods": {},
        }
    result = json.loads(path.read_text())
    if result.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError(f"{path} belongs to a different experiment version")
    return result


def _train_validation_model(
    data: FamilyData,
    method: str,
    columns: np.ndarray,
    names: Sequence[str],
) -> dict[str, object]:
    """Fit on train folds and score all validation rows."""

    results = _load_results(data.family)
    prior = results["methods"].get(method)
    expected_hash = _column_hash(columns)
    if prior is not None:
        if prior["column_hash"] != expected_hash:
            raise ValueError(f"saved method {method} has different columns")
        prediction_path = Path(prior["validation_prediction_path"])
        model_path = Path(prior["model_path"])
        if prediction_path.exists() and model_path.exists():
            print(
                f"{data.family}/{method}: already complete "
                f"(validation TSAUC={prior['validation_tsauc']:.6f})",
                flush=True,
            )
            return prior

    train_rows = np.flatnonzero(data.train_full & data.sampled)
    valid_stop_rows = np.flatnonzero(data.valid_full & data.sampled)
    valid_rows = np.flatnonzero(data.valid_full)
    print(
        f"{data.family}/{method}: materializing {train_rows.size:,} train + "
        f"{valid_stop_rows.size:,} early-stop rows, {columns.size:,} features",
        flush=True,
    )
    started = time.perf_counter()
    x_train = matrix_rows(data.matrix, train_rows, columns)
    x_valid = matrix_rows(data.matrix, valid_stop_rows, columns)
    y_train = data.metadata["target"][train_rows]
    y_valid = data.metadata["target"][valid_stop_rows]
    train_weights = pair_weights_from_reference(
        y_train,
        data.metadata["time_online"][train_rows],
        data.metadata["target"][data.train_full],
        data.metadata["time_online"][data.train_full],
    )
    valid_weights = pair_weights_from_reference(
        y_valid,
        data.metadata["time_online"][valid_stop_rows],
        data.metadata["target"][data.valid_full],
        data.metadata["time_online"][data.valid_full],
    )
    parameters = _parameters(MODEL_SEED)
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weights,
        feature_name=list(names),
        params=parameters,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        weight=valid_weights,
        feature_name=list(names),
        reference=train_set,
        params=parameters,
        free_raw_data=True,
    )
    train_set.construct()
    valid_set.construct()
    del x_train, x_valid, y_train, y_valid, train_weights, valid_weights
    gc.collect()

    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=900,
        valid_sets=[valid_set],
        valid_names=["validation_sample"],
        feval=make_tsauc_evaluator(
            data.metadata["time_online"][valid_stop_rows]
        ),
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    model_path = _family_directory(data.family) / "models" / (
        f"{_safe_name(method)}.txt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    del train_set, valid_set
    gc.collect()

    print(
        f"{data.family}/{method}: predicting {valid_rows.size:,} full "
        "validation rows",
        flush=True,
    )
    prediction = predict_rows(
        booster,
        data.matrix,
        valid_rows,
        columns,
        PREDICTION_CHUNK_ROWS,
    )
    score = tsauc(
        data.metadata["target"][valid_rows],
        prediction,
        data.metadata["time_online"][valid_rows],
    )
    prediction_path = _family_directory(data.family) / "predictions" / (
        f"{_safe_name(method)}.validation.f32.npy"
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, prediction.astype(np.float32))
    elapsed = time.perf_counter() - started
    record = {
        "method": method,
        "feature_count": int(columns.size),
        "column_hash": expected_hash,
        "selection_path": str(_selection_path(data.family, method)),
        "model_path": str(model_path),
        "validation_prediction_path": str(prediction_path),
        "validation_tsauc": float(score),
        "best_iteration": int(booster.best_iteration),
        "train_series": int(np.sum(
            np.isin(
                make_series_folds(
                    data.metadata["online_lengths"],
                    data.metadata["tau_index"],
                    n_splits=5,
                    seed=FOLD_SEED,
                ),
                TRAIN_FOLDS,
            )
        )),
        "validation_series": int(np.sum(
            make_series_folds(
                data.metadata["online_lengths"],
                data.metadata["tau_index"],
                n_splits=5,
                seed=FOLD_SEED,
            ) == VALID_FOLD
        )),
        "train_sample_rows": int(train_rows.size),
        "validation_rows": int(valid_rows.size),
        "elapsed_seconds": float(elapsed),
    }
    results["methods"][method] = record
    _json_dump(_results_path(data.family), results)
    print(
        f"{data.family}/{method}: validation TSAUC={score:.6f}, "
        f"iteration={booster.best_iteration}, elapsed={elapsed / 60:.1f} min",
        flush=True,
    )
    del booster, prediction
    gc.collect()
    return record


def _rank_descending(value: np.ndarray) -> np.ndarray:
    numbers = np.nan_to_num(
        np.asarray(value, dtype=np.float64),
        nan=-np.inf,
        neginf=-np.inf,
        posinf=np.finfo(np.float64).max,
    )
    return np.lexsort((np.arange(numbers.size), -numbers)).astype(np.int32)


def _rank_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(order.size, dtype=np.float64)
    positions[order] = np.arange(order.size, dtype=np.float64)
    return positions


def _screening_rankings(data: FamilyData) -> dict[str, np.ndarray]:
    """Build gain, split, SHAP, consensus, and correlation-filter rankings."""

    directory = _family_directory(data.family) / "rankings"
    directory.mkdir(parents=True, exist_ok=True)
    ranking_path = directory / "rankings.npz"
    if ranking_path.exists():
        saved = np.load(ranking_path, allow_pickle=False)
        return {key: saved[key] for key in saved.files}

    baseline = _load_results(data.family)["methods"].get("all")
    if baseline is None:
        raise RuntimeError("the all-feature screening model must run first")
    booster = lgb.Booster(model_file=baseline["model_path"])
    gain = booster.feature_importance(importance_type="gain").astype(np.float64)
    split = booster.feature_importance(importance_type="split").astype(np.float64)
    gain_order = _rank_descending(gain)
    split_order = _rank_descending(split)

    train_rows = np.flatnonzero(data.train_full & data.sampled)
    rng = np.random.default_rng(81281)
    shap_rows = np.sort(
        rng.choice(train_rows, size=min(SHAP_ROWS, train_rows.size), replace=False)
    )
    print(
        f"{data.family}: computing SHAP ranking on {shap_rows.size:,} "
        "train-only rows",
        flush=True,
    )
    x_shap = matrix_rows(
        data.matrix,
        shap_rows,
        np.arange(len(data.names), dtype=np.int32),
    )
    contribution = booster.predict(
        x_shap,
        pred_contrib=True,
        num_iteration=booster.current_iteration(),
        num_threads=max(1, min(8, os.cpu_count() or 1)),
    )
    shap = np.mean(np.abs(contribution[:, :-1]), axis=0)
    shap_order = _rank_descending(shap)
    del x_shap, contribution
    gc.collect()

    consensus_score = (
        _rank_positions(gain_order)
        + _rank_positions(split_order)
        + _rank_positions(shap_order)
    )
    consensus_order = np.lexsort(
        (np.arange(consensus_score.size), consensus_score)
    ).astype(np.int32)

    pool = gain_order[: min(768, gain_order.size)]
    correlation_rows = np.sort(
        rng.choice(
            train_rows,
            size=min(CORRELATION_ROWS, train_rows.size),
            replace=False,
        )
    )
    print(
        f"{data.family}: correlation-pruning {pool.size:,} gain-ranked "
        f"features on {correlation_rows.size:,} train-only rows",
        flush=True,
    )
    correlation_matrix = matrix_rows(
        data.matrix, correlation_rows, pool
    ).astype(np.float64)
    correlation_matrix -= correlation_matrix.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(correlation_matrix * correlation_matrix, axis=0))
    scale[~np.isfinite(scale) | (scale < 1e-12)] = np.inf
    correlation_matrix /= scale
    correlation = np.abs(correlation_matrix.T @ correlation_matrix)
    np.fill_diagonal(correlation, 0.0)
    kept_local: list[int] = []
    for candidate in range(pool.size):
        if not kept_local or np.max(correlation[candidate, kept_local]) < 0.98:
            kept_local.append(candidate)
    correlation_order = pool[np.asarray(kept_local, dtype=np.int32)]
    del correlation_matrix, correlation
    gc.collect()

    np.savez_compressed(
        ranking_path,
        gain=gain,
        split=split,
        shap=shap,
        gain_order=gain_order,
        split_order=split_order,
        shap_order=shap_order,
        consensus_order=consensus_order,
        gain_nonzero=gain_order[gain[gain_order] > 0],
        correlation_order=correlation_order,
    )
    _json_dump(
        directory / "ranking_summary.json",
        {
            "feature_count": len(data.names),
            "nonzero_gain": int(np.sum(gain > 0)),
            "nonzero_split": int(np.sum(split > 0)),
            "correlation_pool": int(pool.size),
            "correlation_kept_at_abs_lt_0.98": int(correlation_order.size),
            "shap_rows": int(shap_rows.size),
            "correlation_rows": int(correlation_rows.size),
            "all_rows_are_train_only": True,
        },
    )
    del booster
    return {
        "gain": gain,
        "split": split,
        "shap": shap,
        "gain_order": gain_order,
        "split_order": split_order,
        "shap_order": shap_order,
        "consensus_order": consensus_order,
        "gain_nonzero": gain_order[gain[gain_order] > 0],
        "correlation_order": correlation_order,
    }


def _unique_columns(columns: Iterable[int]) -> np.ndarray:
    seen: set[int] = set()
    output: list[int] = []
    for value in columns:
        integer = int(value)
        if integer not in seen:
            seen.add(integer)
            output.append(integer)
    return np.asarray(output, dtype=np.int32)


def _candidate_selections(
    data: FamilyData,
    rankings: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    total = len(data.names)
    candidates: list[tuple[str, np.ndarray]] = [
        ("all", np.arange(total, dtype=np.int32))
    ]
    top_counts = (64, 128, 256, 512, 768)
    for method in ("gain", "split", "shap", "consensus"):
        order = rankings[f"{method}_order"]
        for count in top_counts:
            if count < total:
                candidates.append((f"{method}_top_{count}", order[:count]))

    nonzero = rankings["gain_nonzero"]
    if 0 < nonzero.size < total:
        candidates.append(("gain_nonzero", nonzero))
    correlation = rankings["correlation_order"]
    if 0 < correlation.size < total:
        candidates.append(("gain_corr_0p98", correlation))

    rng = np.random.default_rng(77391 if data.family == "core" else 77392)
    random_count = min(256, total)
    candidates.append(
        (
            f"random_control_{random_count}",
            np.sort(rng.choice(total, size=random_count, replace=False)).astype(
                np.int32
            ),
        )
    )

    slices = bank_slices(data.banks)
    gain = rankings["gain"]
    bank_gain = {
        bank: float(np.sum(gain[slices[bank]])) for bank in data.banks
    }
    ordered_banks = sorted(data.banks, key=lambda bank: (-bank_gain[bank], bank))
    bank_counts = (1, 2, 3, 4, 5) if data.family == "core" else (2, 4, 6, 8, 10)
    for count in bank_counts:
        if count >= len(ordered_banks):
            continue
        selected_banks = ordered_banks[:count]
        columns = _unique_columns(
            column
            for bank in selected_banks
            for column in range(slices[bank].start, slices[bank].stop)
        )
        candidates.append((f"bank_top_{count}", columns))
    _json_dump(
        _family_directory(data.family) / "rankings" / "bank_gain.json",
        {
            "bank_gain": bank_gain,
            "order": ordered_banks,
            "selection_source": "all-feature train-only screening model",
        },
    )

    output: list[tuple[str, np.ndarray]] = []
    seen_hashes: set[str] = set()
    for method, columns in candidates:
        columns = _unique_columns(columns)
        digest = _column_hash(columns)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        output.append((method, columns))
    return output


def run_family(family: str) -> None:
    data = _load_family(family)
    all_columns = np.arange(len(data.names), dtype=np.int32)
    _save_selection(data.family, "all", all_columns, data.names)
    _train_validation_model(data, "all", all_columns, data.names)
    rankings = _screening_rankings(data)
    candidates = _candidate_selections(data, rankings)
    print(
        f"{family}: benchmarking {len(candidates)} unique selection contracts",
        flush=True,
    )
    for method, columns in candidates:
        names = [data.names[int(column)] for column in columns]
        _save_selection(data.family, method, columns, names)
        _train_validation_model(data, method, columns, names)
    results = _load_results(family)
    ordered = sorted(
        results["methods"].values(),
        key=lambda row: (-row["validation_tsauc"], row["feature_count"]),
    )
    eligible = [
        row for row in ordered if _eligible_for_deployment(row["method"])
    ]
    results["ranking"] = [row["method"] for row in ordered]
    results["deployment_eligible_ranking"] = [
        row["method"] for row in eligible
    ]
    results["best_validation_method"] = eligible[0]["method"]
    results["best_validation_tsauc"] = eligible[0]["validation_tsauc"]
    _json_dump(_results_path(family), results)
    print(
        f"{family}: best eligible validation method={eligible[0]['method']} "
        f"TSAUC={eligible[0]['validation_tsauc']:.6f}",
        flush=True,
    )


def _validation_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_family("core")
    rows = np.flatnonzero(data.valid_full)
    return (
        rows,
        data.metadata["target"][rows],
        data.metadata["time_online"][rows],
    )


def _prediction(record: dict[str, object]) -> np.ndarray:
    return np.load(record["validation_prediction_path"], allow_pickle=False)


def freeze_selection(top_methods_per_family: int = 5) -> None:
    frozen_path = ARTIFACT_ROOT / "frozen_selection.json"
    if frozen_path.exists():
        raise FileExistsError(
            f"{frozen_path} already exists; do not silently change a frozen "
            "decision after test evaluation"
        )
    core = _load_results("core")
    candidate = _load_results("candidate")
    if not core["methods"] or not candidate["methods"]:
        raise RuntimeError("run both families before freezing")
    for family_result in (core, candidate):
        if "best_validation_method" not in family_result:
            raise RuntimeError("a family benchmark is incomplete")

    _, target, online_time = _validation_arrays()
    core_rows = sorted(
        (
            row for row in core["methods"].values()
            if _eligible_for_deployment(row["method"])
        ),
        key=lambda row: -row["validation_tsauc"],
    )[:top_methods_per_family]
    candidate_rows = sorted(
        (
            row for row in candidate["methods"].values()
            if _eligible_for_deployment(row["method"])
        ),
        key=lambda row: -row["validation_tsauc"],
    )[:top_methods_per_family]
    weights = np.linspace(0.0, 1.0, 21)
    search: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    prediction_cache: dict[tuple[str, str], np.ndarray] = {}
    for core_row in core_rows:
        core_prediction = _prediction(core_row)
        prediction_cache[("core", core_row["method"])] = core_prediction
        for candidate_row in candidate_rows:
            candidate_prediction = _prediction(candidate_row)
            prediction_cache[("candidate", candidate_row["method"])] = (
                candidate_prediction
            )
            core_margin = _safe_logit(core_prediction)
            candidate_margin = _safe_logit(candidate_prediction)
            for core_weight in weights:
                prediction = _logistic(
                    core_weight * core_margin
                    + (1.0 - core_weight) * candidate_margin
                )
                score = tsauc(target, prediction, online_time)
                record = {
                    "core_method": core_row["method"],
                    "candidate_method": candidate_row["method"],
                    "core_weight": float(core_weight),
                    "candidate_weight": float(1.0 - core_weight),
                    "validation_tsauc": float(score),
                }
                search.append(record)
                if best is None or (
                    score,
                    -abs(float(core_weight) - 0.5),
                ) > (
                    best["validation_tsauc"],
                    -abs(best["core_weight"] - 0.5),
                ):
                    best = record
    assert best is not None

    baseline_core = core["methods"]["all"]
    baseline_candidate = candidate["methods"]["all"]
    baseline_prediction = _logistic(
        0.5 * _safe_logit(_prediction(baseline_core))
        + 0.5 * _safe_logit(_prediction(baseline_candidate))
    )
    baseline_score = tsauc(target, baseline_prediction, online_time)
    selected_core = core["methods"][best["core_method"]]
    selected_candidate = candidate["methods"][best["candidate_method"]]
    contract = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_without_test_labels": True,
        "selection_split": "fold 1 validation; folds 2-4 training",
        "locked_test_split": "fold 0",
        "search_space": {
            "top_methods_per_family": top_methods_per_family,
            "negative_controls_are_deployment_ineligible": True,
            "core_methods": [row["method"] for row in core_rows],
            "candidate_methods": [row["method"] for row in candidate_rows],
            "core_weight_grid": weights.tolist(),
        },
        "selected": {
            **best,
            "core_feature_count": selected_core["feature_count"],
            "candidate_feature_count": selected_candidate["feature_count"],
            "core_best_iteration": selected_core["best_iteration"],
            "candidate_best_iteration": selected_candidate["best_iteration"],
            "core_column_hash": selected_core["column_hash"],
            "candidate_column_hash": selected_candidate["column_hash"],
        },
        "all_feature_baseline": {
            "core_method": "all",
            "candidate_method": "all",
            "core_weight": 0.5,
            "candidate_weight": 0.5,
            "validation_tsauc": float(baseline_score),
            "core_best_iteration": baseline_core["best_iteration"],
            "candidate_best_iteration": baseline_candidate["best_iteration"],
        },
        "validation_gain_over_all_features": float(
            best["validation_tsauc"] - baseline_score
        ),
        "contract_hash": "",
    }
    contract["contract_hash"] = _sha256_json(
        {key: value for key, value in contract.items() if key != "contract_hash"}
    )
    _json_dump(ARTIFACT_ROOT / "ensemble_validation_search.json", search)
    _json_dump(frozen_path, contract)
    print(json.dumps(contract, indent=2), flush=True)


def _train_fixed_model(
    data: FamilyData,
    method: str,
    rounds: int,
    tag: str,
) -> tuple[lgb.Booster, np.ndarray]:
    columns, names = _load_selection(data.family, method)
    development_rows = np.flatnonzero(data.development_full & data.sampled)
    print(
        f"final {tag}: materializing {development_rows.size:,} development "
        f"rows x {columns.size:,} features",
        flush=True,
    )
    matrix = matrix_rows(data.matrix, development_rows, columns)
    target = data.metadata["target"][development_rows]
    weights = pair_weights_from_reference(
        target,
        data.metadata["time_online"][development_rows],
        data.metadata["target"][data.development_full],
        data.metadata["time_online"][data.development_full],
    )
    parameters = _parameters(MODEL_SEED + 1000)
    dataset = lgb.Dataset(
        matrix,
        label=target,
        weight=weights,
        feature_name=names,
        params=parameters,
        free_raw_data=True,
    )
    dataset.construct()
    del matrix, target, weights
    gc.collect()
    booster = lgb.train(parameters, dataset, num_boost_round=int(rounds))
    del dataset
    gc.collect()
    model_path = ARTIFACT_ROOT / "locked_test_models" / f"{tag}.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    return booster, columns


def _bootstrap_subfold_se(
    target: np.ndarray,
    prediction: np.ndarray,
    online_time: np.ndarray,
    series_position: np.ndarray,
    n_groups: int = 5,
) -> dict[str, object]:
    """Series-cluster subfold dispersion without row-level pseudo-replication."""

    unique = np.unique(series_position)
    group_by_series = {
        int(series): index % n_groups for index, series in enumerate(unique)
    }
    group = np.asarray(
        [group_by_series[int(series)] for series in series_position],
        dtype=np.int8,
    )
    scores = np.asarray(
        [
            tsauc(target[group == index], prediction[group == index],
                  online_time[group == index])
            for index in range(n_groups)
        ],
        dtype=np.float64,
    )
    return {
        "series_subfold_scores": scores.tolist(),
        "series_subfold_mean": float(scores.mean()),
        "series_subfold_standard_deviation": float(scores.std(ddof=1)),
        "series_subfold_standard_error": float(scores.std(ddof=1) / math.sqrt(n_groups)),
        "note": (
            "Dispersion across five deterministic groups within the locked "
            "test series; it is not a row-level standard error."
        ),
    }


def evaluate_locked_test() -> None:
    report_path = ARTIFACT_ROOT / "locked_test_results.json"
    if report_path.exists():
        raise FileExistsError(
            f"{report_path} already exists; the locked test may not be "
            "reopened during this experiment"
        )
    frozen_path = ARTIFACT_ROOT / "frozen_selection.json"
    if not frozen_path.exists():
        raise RuntimeError("freeze selections before opening the locked test")
    frozen = json.loads(frozen_path.read_text())
    expected_hash = _sha256_json(
        {key: value for key, value in frozen.items() if key != "contract_hash"}
    )
    if frozen["contract_hash"] != expected_hash:
        raise ValueError("frozen selection contract has been modified")

    core = _load_family("core")
    candidate = _load_family("candidate")
    if not np.array_equal(core.row_fold, candidate.row_fold):
        raise AssertionError("family split assignments differ")
    test_rows = np.flatnonzero(core.test_full)

    configurations = {
        "selected": frozen["selected"],
        "all_feature_baseline": frozen["all_feature_baseline"],
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    trained: dict[tuple[str, str, int], tuple[lgb.Booster, np.ndarray]] = {}
    for configuration_name, configuration in configurations.items():
        predictions[configuration_name] = {}
        for family, data in (("core", core), ("candidate", candidate)):
            method = configuration[f"{family}_method"]
            rounds = int(configuration[f"{family}_best_iteration"])
            key = (family, method, rounds)
            if key not in trained:
                trained[key] = _train_fixed_model(
                    data,
                    method,
                    rounds,
                    f"{family}__{_safe_name(method)}__r{rounds}",
                )
            booster, columns = trained[key]
            print(
                f"locked test: predicting {configuration_name}/{family} on "
                f"{test_rows.size:,} rows",
                flush=True,
            )
            predictions[configuration_name][family] = predict_rows(
                booster,
                data.matrix,
                test_rows,
                columns,
                PREDICTION_CHUNK_ROWS,
            )

    # This is the first point in the workflow that indexes fold-0 targets.
    target = core.metadata["target"][test_rows]
    online_time = core.metadata["time_online"][test_rows]
    series_position = core.metadata["series_position"][test_rows]
    report: dict[str, object] = {
        "experiment_version": EXPERIMENT_VERSION,
        "frozen_contract_hash": frozen["contract_hash"],
        "test_opened_after_freeze": True,
        "test_fold": TEST_FOLD,
        "test_series": int(np.unique(series_position).size),
        "test_rows": int(test_rows.size),
        "configurations": {},
    }
    for configuration_name, configuration in configurations.items():
        core_prediction = predictions[configuration_name]["core"]
        candidate_prediction = predictions[configuration_name]["candidate"]
        core_weight = float(configuration["core_weight"])
        ensemble = _logistic(
            core_weight * _safe_logit(core_prediction)
            + (1.0 - core_weight) * _safe_logit(candidate_prediction)
        )
        result = {
            "core_method": configuration["core_method"],
            "candidate_method": configuration["candidate_method"],
            "core_weight": core_weight,
            "candidate_weight": 1.0 - core_weight,
            "core_tsauc": float(tsauc(target, core_prediction, online_time)),
            "candidate_tsauc": float(
                tsauc(target, candidate_prediction, online_time)
            ),
            "ensemble_tsauc": float(tsauc(target, ensemble, online_time)),
            **_bootstrap_subfold_se(
                target,
                ensemble,
                online_time,
                series_position,
            ),
        }
        report["configurations"][configuration_name] = result
        prediction_path = ARTIFACT_ROOT / (
            f"{configuration_name}.locked_test_prediction.f32.npy"
        )
        np.save(prediction_path, ensemble.astype(np.float32))
    report["selected_minus_all_feature_test_tsauc"] = float(
        report["configurations"]["selected"]["ensemble_tsauc"]
        - report["configurations"]["all_feature_baseline"]["ensemble_tsauc"]
    )
    paired_delta = np.asarray(
        report["configurations"]["selected"]["series_subfold_scores"],
        dtype=np.float64,
    ) - np.asarray(
        report["configurations"]["all_feature_baseline"][
            "series_subfold_scores"
        ],
        dtype=np.float64,
    )
    paired_se = float(paired_delta.std(ddof=1) / math.sqrt(paired_delta.size))
    paired_mean = float(paired_delta.mean())
    # Five deterministic test-series groups imply four degrees of freedom.
    critical = 2.7764451051977987
    report["paired_series_subfold_comparison"] = {
        "selected_minus_baseline_by_subfold": paired_delta.tolist(),
        "mean_difference": paired_mean,
        "standard_error": paired_se,
        "95_percent_t_interval": [
            paired_mean - critical * paired_se,
            paired_mean + critical * paired_se,
        ],
        "conclusion": (
            "Feature selection is not a statistically resolved improvement "
            "when the interval includes zero."
        ),
    }
    _json_dump(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    family = subparsers.add_parser("run-family")
    family.add_argument("--family", choices=tuple(FAMILY_CACHE), required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--top-methods-per-family", type=int, default=5)
    subparsers.add_parser("evaluate-test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-family":
        run_family(args.family)
    elif args.command == "freeze":
        freeze_selection(args.top_methods_per_family)
    elif args.command == "evaluate-test":
        evaluate_locked_test()
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
