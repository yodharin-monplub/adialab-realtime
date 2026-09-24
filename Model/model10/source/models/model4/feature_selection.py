"""Leakage-safe nested feature selection shared by notebook experiments.

Detector banks are frozen first. This module then ranks their joint feature
union independently inside each development fold, benchmarks several ranking
methods and feature counts on untouched scoring folds, freezes exactly one
winner, and evaluates only that winner on the sealed outer holdout.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from models.model4 import model4_pipeline as base


FEATURE_SELECTION_VERSION = "model4-nested-feature-selection-v1"
SUPPORTED_METHODS = ("gain", "split", "consensus", "gain_corr")


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """All settings that can change the feature-search result."""

    methods: tuple[str, ...] = SUPPORTED_METHODS
    feature_counts: tuple[int, ...] = (32, 64, 128, 256, 512)
    num_boost_round: int = 750
    early_stopping_rounds: int = 75
    model_seed: int = 106_031
    correlation_pool_size: int = 1_024
    correlation_sample_rows: int = 4_096
    correlation_abs_threshold: float = 0.98
    ranking_seed: int = 106_013

    def validate(self) -> None:
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("feature-selection methods must be unique and non-empty")
        unknown = set(self.methods) - set(SUPPORTED_METHODS)
        if unknown:
            raise ValueError(f"unsupported feature-selection methods: {sorted(unknown)}")
        if (
            not self.feature_counts
            or tuple(sorted(set(self.feature_counts))) != self.feature_counts
            or any(count < 1 for count in self.feature_counts)
        ):
            raise ValueError("feature_counts must be unique, increasing positive integers")
        if min(self.num_boost_round, self.early_stopping_rounds) < 1:
            raise ValueError("boosting limits must be positive")
        if min(self.correlation_pool_size, self.correlation_sample_rows) < 2:
            raise ValueError("correlation pool and row sample must be at least two")
        if not 0.0 < self.correlation_abs_threshold < 1.0:
            raise ValueError("correlation_abs_threshold must be in (0, 1)")

    def protocol(
        self, workflow_version: str = FEATURE_SELECTION_VERSION
    ) -> dict[str, Any]:
        return {
            "feature_selection_version": workflow_version,
            **asdict(self),
        }


def _root(model_config: base.Model4Config) -> Path:
    return model_config.artifact_path / "feature_selection"


def _workflow_version(model_config: base.Model4Config) -> str:
    """Use a model-specific label while preserving the Model 4 contracts."""

    return str(
        getattr(
            model_config,
            "feature_selection_workflow_version",
            FEATURE_SELECTION_VERSION,
        )
    )


def _rank_descending(values: np.ndarray) -> np.ndarray:
    numbers = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=-np.inf,
        neginf=-np.inf,
        posinf=np.finfo(np.float64).max,
    )
    return np.lexsort((np.arange(numbers.size), -numbers)).astype(np.int32)


def _rank_positions(order: np.ndarray) -> np.ndarray:
    order = np.asarray(order, dtype=np.int32)
    positions = np.empty(order.size, dtype=np.float64)
    positions[order] = np.arange(order.size, dtype=np.float64)
    return positions


def _validate_order(order: np.ndarray, feature_count: int, label: str) -> None:
    order = np.asarray(order)
    if order.shape != (feature_count,):
        raise ValueError(f"{label} ranking has the wrong length")
    if not np.array_equal(np.sort(order), np.arange(feature_count)):
        raise ValueError(f"{label} ranking is not a feature permutation")


def _source_selection(
    model_config: base.Model4Config,
) -> tuple[np.ndarray, list[str], tuple[str, ...], dict[str, Any]]:
    columns, names, banks = base.load_selection_arrays(model_config, "joint")
    selection_path = model_config.artifact_path / "selected_detectors.json"
    if not selection_path.exists():
        raise FileNotFoundError("freeze detector banks before feature selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("joint_column_hash") != base._column_hash(columns):
        raise ValueError("detector selection JSON and joint NPZ columns disagree")
    return columns, names, banks, selection


def _correlation_pruned_order(
    data: base.ExperimentData,
    train_mask: np.ndarray,
    source_columns: np.ndarray,
    gain_order: np.ndarray,
    config: FeatureSelectionConfig,
    fold: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Prioritize non-redundant gain features using training-role rows only."""

    pool_size = min(config.correlation_pool_size, gain_order.size)
    pool_local = gain_order[:pool_size]
    available_rows = np.flatnonzero(train_mask)
    rng = np.random.default_rng(config.ranking_seed + fold)
    sample_rows = np.sort(
        rng.choice(
            available_rows,
            size=min(config.correlation_sample_rows, available_rows.size),
            replace=False,
        )
    )
    matrix = base.matrix_rows(
        data.joint_matrix,
        sample_rows,
        source_columns[pool_local],
    ).astype(np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(matrix * matrix, axis=0))
    scale[~np.isfinite(scale) | (scale < 1e-12)] = np.inf
    matrix /= scale
    correlation = np.abs(matrix.T @ matrix)
    np.fill_diagonal(correlation, 0.0)
    kept_pool_positions: list[int] = []
    for candidate in range(pool_size):
        if (
            not kept_pool_positions
            or float(np.max(correlation[candidate, kept_pool_positions]))
            < config.correlation_abs_threshold
        ):
            kept_pool_positions.append(candidate)
    kept = pool_local[np.asarray(kept_pool_positions, dtype=np.int32)]
    used = np.zeros(gain_order.size, dtype=bool)
    used[kept] = True
    order = np.concatenate((kept, gain_order[~used[gain_order]])).astype(np.int32)
    del matrix, correlation
    gc.collect()
    _validate_order(order, gain_order.size, "gain_corr")
    return order, {
        "correlation_rows": int(sample_rows.size),
        "correlation_pool": int(pool_size),
        "nonredundant_features_in_pool": int(kept.size),
    }


def _fold_rankings(
    data: base.ExperimentData,
    split: base.SplitAssignment,
    model_config: base.Model4Config,
    selection_config: FeatureSelectionConfig,
    fold: int,
    source_columns: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load or create rankings whose source model cannot see the score fold."""

    directory = _root(model_config) / "rankings"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"fold_{fold}.npz"
    summary_path = directory / f"fold_{fold}.json"
    train_mask, stop_mask, score_mask, stop_fold = base.development_fold_roles(
        split, fold, model_config.development_folds
    )
    baseline_path = (
        model_config.artifact_path / "combined_cv" / "joint" / f"fold_{fold}.txt"
    )
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"joint baseline fold is missing: {baseline_path}; run baseline CV first"
        )
    contract = {
        "feature_selection_version": _workflow_version(model_config),
        "fold": fold,
        "early_stop_fold": stop_fold,
        "source_column_hash": base._column_hash(source_columns),
        "source_feature_count": int(source_columns.size),
        "baseline_model": str(baseline_path.relative_to(model_config.artifact_path)),
        "methods": list(selection_config.methods),
        "ranking_seed": selection_config.ranking_seed,
        "correlation_pool_size": selection_config.correlation_pool_size,
        "correlation_sample_rows": selection_config.correlation_sample_rows,
        "correlation_abs_threshold": selection_config.correlation_abs_threshold,
        "ranking_model_fit_rows": "training role only",
        "ranking_model_early_stopping_rows": "separate stop fold",
        "score_fold_excluded_from_fit_early_stop_and_ranking": True,
        "correlation_rows_use_training_role_only": True,
    }
    contract["configuration_hash"] = base._sha_json(contract)
    base._write_or_validate_json(summary_path, contract)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            rankings = {key: saved[key].astype(np.int32) for key in saved.files}
        for method in SUPPORTED_METHODS:
            _validate_order(rankings[method], source_columns.size, method)
        return rankings

    booster = lgb.Booster(model_file=str(baseline_path))
    gain = booster.feature_importance("gain").astype(np.float64)
    split_importance = booster.feature_importance("split").astype(np.float64)
    if gain.size != source_columns.size or split_importance.size != source_columns.size:
        raise ValueError("joint baseline feature order differs from detector selection")
    gain_order = _rank_descending(gain)
    split_order = _rank_descending(split_importance)
    consensus_score = _rank_positions(gain_order) + _rank_positions(split_order)
    consensus_order = np.lexsort(
        (np.arange(consensus_score.size), consensus_score)
    ).astype(np.int32)
    gain_corr_order, correlation_summary = _correlation_pruned_order(
        data,
        train_mask,
        source_columns,
        gain_order,
        selection_config,
        fold,
    )
    rankings = {
        "gain": gain_order,
        "split": split_order,
        "consensus": consensus_order,
        "gain_corr": gain_corr_order,
    }
    for method, order in rankings.items():
        _validate_order(order, source_columns.size, method)
    np.savez_compressed(path, **rankings)
    ranking_report = {
        **contract,
        **correlation_summary,
        "train_series_rows": int(np.sum(train_mask)),
        "early_stop_rows_excluded": int(np.sum(stop_mask)),
        "score_rows_excluded": int(np.sum(score_mask)),
    }
    base._json_dump(directory / f"fold_{fold}.report.json", ranking_report)
    del booster, gain, split_importance
    gc.collect()
    return rankings


def build_feature_rankings(
    data: base.ExperimentData,
    split: base.SplitAssignment,
    model_config: base.Model4Config,
    selection_config: FeatureSelectionConfig,
) -> list[dict[str, np.ndarray]]:
    """Build one train-only ranking set for every untouched scoring fold."""

    selection_config.validate()
    source_columns, _, _, _ = _source_selection(model_config)
    if max(selection_config.feature_counts) > source_columns.size:
        raise ValueError(
            "largest requested feature count exceeds the selected detector union"
        )
    return [
        _fold_rankings(
            data,
            split,
            model_config,
            selection_config,
            fold,
            source_columns,
        )
        for fold in range(model_config.development_folds)
    ]


def _candidate_directory(
    model_config: base.Model4Config, method: str, count: int
) -> Path:
    return _root(model_config) / "candidates" / method / f"top_{count}"


def _evaluate_candidate(
    data: base.ExperimentData,
    split: base.SplitAssignment,
    model_config: base.Model4Config,
    selection_config: FeatureSelectionConfig,
    rankings: Sequence[dict[str, np.ndarray]],
    source_columns: np.ndarray,
    source_names: Sequence[str],
    method: str,
    count: int,
) -> dict[str, Any]:
    """Nestedly rank, fit, early-stop, and score one method/count candidate."""

    directory = _candidate_directory(model_config, method, count)
    directory.mkdir(parents=True, exist_ok=True)
    development_rows = split.development_rows
    local_fold = split.row_inner_fold[development_rows]
    oof = np.full(development_rows.size, np.nan, dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    method_offset = selection_config.methods.index(method) * 10_000

    for fold in range(model_config.development_folds):
        train_full, stop_full, score_full, stop_fold = base.development_fold_roles(
            split, fold, model_config.development_folds
        )
        score_rows = np.flatnonzero(score_full)
        train_rows = np.flatnonzero(train_full & split.sampled_row_mask)
        stop_rows = np.flatnonzero(stop_full & split.sampled_row_mask)
        local_columns = rankings[fold][method][:count]
        columns = source_columns[local_columns]
        names = [source_names[int(index)] for index in local_columns]
        fold_contract = {
            "feature_selection_version": _workflow_version(model_config),
            "method": method,
            "feature_count": count,
            "fold": fold,
            "early_stop_fold": stop_fold,
            "columns": columns.tolist(),
            "column_hash": base._column_hash(columns),
            "ranking_uses_training_role_only": True,
            "model_seed": selection_config.model_seed + method_offset + 100 * count + fold,
            "num_boost_round": selection_config.num_boost_round,
            "early_stopping_rounds": selection_config.early_stopping_rounds,
        }
        fold_contract["configuration_hash"] = base._sha_json(fold_contract)
        base._write_or_validate_json(
            directory / f"fold_{fold}.contract.json", fold_contract
        )
        model_path = directory / f"fold_{fold}.txt"
        prediction_path = directory / f"fold_{fold}.prediction.f32.npy"
        result_path = directory / f"fold_{fold}.json"
        if model_path.exists() and prediction_path.exists() and result_path.exists():
            prediction = base._load_saved_prediction(prediction_path, score_rows.size)
            record = json.loads(result_path.read_text(encoding="utf-8"))
            print(
                f"feature search {method}/{count} fold {fold}: already complete, "
                f"TSAUC={record['tsauc']:.6f}",
                flush=True,
            )
        else:
            print(
                f"feature search {method}/{count} fold {fold}: "
                f"{train_rows.size:,} train, {stop_rows.size:,} stop, "
                f"{score_rows.size:,} score rows",
                flush=True,
            )
            started = time.perf_counter()
            parameters = base._seeded_parameters(
                model_config,
                int(fold_contract["model_seed"]),
            )
            booster = base._train_early_stopped(
                matrix=data.joint_matrix,
                metadata=data.metadata,
                train_rows=train_rows,
                validation_rows=stop_rows,
                train_reference_mask=train_full,
                validation_reference_mask=stop_full,
                columns=columns,
                names=names,
                parameters=parameters,
                num_boost_round=selection_config.num_boost_round,
                early_stopping_rounds=selection_config.early_stopping_rounds,
                log_every=model_config.log_every,
                metric="tsauc",
            )
            booster.save_model(model_path, num_iteration=booster.best_iteration)
            prediction = base.predict_rows(
                booster,
                data.joint_matrix,
                score_rows,
                columns,
                model_config.prediction_chunk_rows,
            )
            np.save(prediction_path, prediction.astype(np.float32))
            target = data.metadata["target"][score_rows]
            record = {
                "fold": fold,
                "early_stop_fold": stop_fold,
                "tsauc": float(
                    base.tsauc(
                        target,
                        prediction,
                        data.metadata["time_online"][score_rows],
                    )
                ),
                "roc_auc_diagnostic": float(roc_auc_score(target, prediction)),
                "best_iteration": int(booster.best_iteration),
                "train_sample_rows": int(train_rows.size),
                "early_stop_sample_rows": int(stop_rows.size),
                "full_score_rows": int(score_rows.size),
                "elapsed_seconds": float(time.perf_counter() - started),
                "column_hash": fold_contract["column_hash"],
            }
            base._json_dump(result_path, record)
            del booster, target
            gc.collect()
        expected_rows = development_rows[local_fold == fold]
        if not np.array_equal(expected_rows, score_rows):
            raise AssertionError("development OOF row order changed")
        oof[local_fold == fold] = prediction
        fold_records.append(record)

    if not np.all(np.isfinite(oof)):
        raise AssertionError(f"incomplete feature-search OOF: {method}/{count}")
    fold_scores = [float(record["tsauc"]) for record in fold_records]
    target = data.metadata["target"][development_rows]
    statistics = {
        **base._mean_and_se(fold_scores),
        "pooled_development_oof": float(
            base.tsauc(
                target,
                oof,
                data.metadata["time_online"][development_rows],
            )
        ),
    }
    result = {
        "method": method,
        "feature_count": int(count),
        "tsauc": statistics,
        "best_iterations": [
            int(record["best_iteration"]) for record in fold_records
        ],
        "fixed_refit_rounds": max(
            1,
            int(round(float(np.median([record["best_iteration"] for record in fold_records])))),
        ),
        "fold_column_hashes": [record["column_hash"] for record in fold_records],
    }
    np.save(directory / "development_oof.f32.npy", oof)
    base._json_dump(directory / "summary.json", result)
    return result


def choose_feature_winner(
    candidates: Sequence[dict[str, Any]], methods: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the predeclared TSAUC/SE/size/method tie-break order."""

    if not candidates:
        raise ValueError("feature search produced no candidates")
    method_position = {method: index for index, method in enumerate(methods)}
    if len(method_position) != len(methods):
        raise ValueError("feature-selection method order must be unique")
    unknown = {row["method"] for row in candidates} - set(method_position)
    if unknown:
        raise ValueError(f"candidate methods are not configured: {sorted(unknown)}")
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["tsauc"]["fold_mean"]),
            float(row["tsauc"]["fold_standard_error"]),
            int(row["feature_count"]),
            method_position[row["method"]],
        ),
    )
    return ordered[0], ordered


def search_feature_subsets(
    data: base.ExperimentData,
    split: base.SplitAssignment,
    baseline_joint: dict[str, Any],
    model_config: base.Model4Config,
    selection_config: FeatureSelectionConfig,
) -> dict[str, Any]:
    """Benchmark all configured methods/counts and choose one development winner."""

    selection_config.validate()
    source_columns, source_names, source_banks, detector_selection = (
        _source_selection(model_config)
    )
    if baseline_joint.get("column_hash") != base._column_hash(source_columns):
        raise ValueError("joint baseline and frozen detector columns differ")
    rankings = build_feature_rankings(data, split, model_config, selection_config)
    manifest = {
        "feature_selection_version": _workflow_version(model_config),
        "model_protocol_hash": base._sha_json(model_config.protocol()),
        "feature_selection_protocol": selection_config.protocol(
            _workflow_version(model_config)
        ),
        "source_detector_selection_hash": detector_selection["selection_hash"],
        "source_column_hash": base._column_hash(source_columns),
        "source_feature_count": int(source_columns.size),
        "source_banks": list(source_banks),
        "selection_roles": (
            "ranking=train folds only; early stopping=next fold; "
            "candidate score=untouched held fold"
        ),
        "holdout_used": False,
    }
    manifest["configuration_hash"] = base._sha_json(manifest)
    base._write_or_validate_json(_root(model_config) / "manifest.json", manifest)
    completed_path = _root(model_config) / "search_results.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("configuration_hash") != manifest["configuration_hash"]:
            raise ValueError("saved feature search belongs to another configuration")
        return completed

    candidates = [
        _evaluate_candidate(
            data,
            split,
            model_config,
            selection_config,
            rankings,
            source_columns,
            source_names,
            method,
            count,
        )
        for method in selection_config.methods
        for count in selection_config.feature_counts
    ]
    winner, ordered = choose_feature_winner(
        candidates, selection_config.methods
    )
    result = {
        "feature_selection_version": _workflow_version(model_config),
        "configuration_hash": manifest["configuration_hash"],
        "winner_rule": (
            "largest development-fold mean TSAUC; then smaller SE, fewer features, "
            "and configured method order"
        ),
        "winner": winner,
        "ranking": ordered,
        "all_feature_baseline": baseline_joint,
        "holdout_used": False,
    }
    base._json_dump(completed_path, result)
    return result


def feature_search_table(search: dict[str, Any]) -> pd.DataFrame:
    """Return a compact notebook table ordered by the frozen winner rule."""

    return pd.DataFrame.from_records(
        [
            {
                "method": row["method"],
                "features": row["feature_count"],
                "mean_TSAUC": row["tsauc"]["fold_mean"],
                "TSAUC_SE": row["tsauc"]["fold_standard_error"],
                "pooled_TSAUC": row["tsauc"]["pooled_development_oof"],
                "fixed_rounds": row["fixed_refit_rounds"],
            }
            for row in search["ranking"]
        ]
    )


def _aggregate_order(
    rankings: Sequence[dict[str, np.ndarray]], method: str
) -> np.ndarray:
    """Borda-aggregate all nested rankings for final development fitting."""

    feature_count = rankings[0][method].size
    score = np.zeros(feature_count, dtype=np.float64)
    for fold_ranking in rankings:
        order = fold_ranking[method]
        _validate_order(order, feature_count, method)
        score += _rank_positions(order)
    return np.lexsort((np.arange(feature_count), score)).astype(np.int32)


def freeze_feature_winner(
    data: base.ExperimentData,
    split: base.SplitAssignment,
    search: dict[str, Any],
    model_config: base.Model4Config,
    selection_config: FeatureSelectionConfig,
) -> dict[str, Any]:
    """Freeze one method/count and its development-only aggregate feature order."""

    manifest_path = _root(model_config) / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("complete feature search before freezing a winner")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if search.get("configuration_hash") != manifest.get("configuration_hash"):
        raise ValueError("feature search and manifest configuration hashes differ")
    source_columns, source_names, _, detector_selection = _source_selection(
        model_config
    )
    rankings = build_feature_rankings(data, split, model_config, selection_config)
    winner = search["winner"]
    method = str(winner["method"])
    count = int(winner["feature_count"])
    aggregate_order = _aggregate_order(rankings, method)
    local_columns = aggregate_order[:count]
    columns = source_columns[local_columns]
    names = [source_names[int(index)] for index in local_columns]
    contract = {
        "feature_selection_version": _workflow_version(model_config),
        "created_without_holdout_scoring": True,
        "configuration_hash": manifest["configuration_hash"],
        "source_detector_selection_hash": detector_selection["selection_hash"],
        "winner_rule": search["winner_rule"],
        "method": method,
        "feature_count": count,
        "development_tsauc": winner["tsauc"],
        "fixed_refit_rounds": int(winner["fixed_refit_rounds"]),
        "final_ranking": "Borda aggregation of all nested fold rankings",
        "columns": columns.tolist(),
        "column_hash": base._column_hash(columns),
        "feature_names": names,
        "holdout_used": False,
    }
    contract["contract_hash"] = base._sha_json(contract)
    base._write_or_validate_json(_root(model_config) / "frozen_winner.json", contract)
    selection_path = _root(model_config) / "winning_features.npz"
    if selection_path.exists():
        with np.load(selection_path, allow_pickle=False) as saved:
            if not (
                np.array_equal(saved["columns"], columns)
                and saved["names"].tolist() == names
            ):
                raise ValueError("saved winning feature NPZ differs from frozen winner")
    else:
        np.savez_compressed(
            selection_path,
            columns=columns.astype(np.int32),
            names=np.asarray(names, dtype=np.str_),
            method=np.asarray(method),
            feature_count=np.asarray(count, dtype=np.int32),
        )
    return contract


def _validated_winner(
    supplied: dict[str, Any], model_config: base.Model4Config
) -> dict[str, Any]:
    path = _root(model_config) / "frozen_winner.json"
    if not path.exists():
        raise RuntimeError("freeze the feature-selection winner first")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if base._json_ready(supplied) != saved:
        raise ValueError("supplied feature winner differs from the frozen disk contract")
    unhashed = dict(saved)
    claimed = unhashed.pop("contract_hash", None)
    if claimed != base._sha_json(unhashed):
        raise ValueError("frozen feature-selection contract hash is invalid")
    return saved


def fit_feature_selected_model(
    data: base.ExperimentData,
    split: base.SplitAssignment,
    frozen: dict[str, Any],
    model_config: base.Model4Config,
) -> dict[str, Any]:
    """Fit the frozen feature subset and predict holdout without reading labels."""

    frozen = _validated_winner(frozen, model_config)
    columns = np.asarray(frozen["columns"], dtype=np.int32)
    names = list(frozen["feature_names"])
    directory = _root(model_config) / "final"
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.txt"
    prediction_path = directory / "holdout_prediction.f32.npy"
    fit_contract = {
        "feature_selection_version": _workflow_version(model_config),
        "frozen_contract_hash": frozen["contract_hash"],
        "rounds": int(frozen["fixed_refit_rounds"]),
        "feature_count": int(columns.size),
        "column_hash": base._column_hash(columns),
        "model_seed": int(model_config.combined_model_seed + 40_000),
        "train_sample_fraction": model_config.train_sample_fraction,
        "row_sample_seed": model_config.row_sample_seed,
        "holdout_used_for_training_or_early_stopping": False,
    }
    fit_contract["configuration_hash"] = base._sha_json(fit_contract)
    base._write_or_validate_json(directory / "fit_contract.json", fit_contract)

    if not model_path.exists():
        train_full = split.development_row_mask
        train_rows = np.flatnonzero(train_full & split.sampled_row_mask)
        print(
            f"feature winner final fit: {train_rows.size:,} sampled development "
            f"rows x {columns.size:,} features, {fit_contract['rounds']} rounds",
            flush=True,
        )
        matrix = base.matrix_rows(data.joint_matrix, train_rows, columns)
        target = data.metadata["target"][train_rows]
        weight = base.pair_weights_from_reference(
            target,
            data.metadata["time_online"][train_rows],
            data.metadata["target"][train_full],
            data.metadata["time_online"][train_full],
        )
        parameters = base._seeded_parameters(
            model_config, int(fit_contract["model_seed"])
        )
        dataset = lgb.Dataset(
            matrix,
            label=target,
            weight=weight,
            feature_name=names,
            params=parameters,
            free_raw_data=True,
        )
        dataset.construct()
        del matrix, target, weight
        gc.collect()
        booster = lgb.train(
            parameters,
            dataset,
            num_boost_round=int(fit_contract["rounds"]),
            callbacks=[lgb.log_evaluation(period=model_config.log_every)],
        )
        booster.save_model(model_path, num_iteration=int(fit_contract["rounds"]))
        del booster, dataset
        gc.collect()

    holdout_rows = split.holdout_rows
    if prediction_path.exists():
        prediction = base._load_saved_prediction(prediction_path, holdout_rows.size)
    else:
        booster = lgb.Booster(model_file=str(model_path))
        prediction = base.predict_rows(
            booster,
            data.joint_matrix,
            holdout_rows,
            columns,
            model_config.prediction_chunk_rows,
        )
        np.save(prediction_path, prediction.astype(np.float32))
        del booster
        gc.collect()
    manifest = {
        "feature_selection_version": _workflow_version(model_config),
        "frozen_contract_hash": frozen["contract_hash"],
        "model": str(model_path.relative_to(model_config.artifact_path)),
        "prediction": str(prediction_path.relative_to(model_config.artifact_path)),
        "holdout_rows": int(holdout_rows.size),
        "holdout_labels_read": False,
    }
    base._write_or_validate_json(directory / "prediction_manifest.json", manifest)
    return manifest


def evaluate_feature_selected_holdout(
    data: base.ExperimentData,
    split: base.SplitAssignment,
    frozen: dict[str, Any],
    model_config: base.Model4Config,
) -> dict[str, Any]:
    """Score only the frozen feature winner on the sealed holdout once."""

    frozen = _validated_winner(frozen, model_config)
    result_path = _root(model_config) / "holdout_results.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("frozen_contract_hash") != frozen["contract_hash"]:
            raise ValueError("saved feature holdout result belongs to another winner")
        print("feature-selected holdout was already evaluated; reusing it", flush=True)
        return saved
    manifest_path = _root(model_config) / "final" / "prediction_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("fit the frozen feature winner before holdout evaluation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("frozen_contract_hash") != frozen["contract_hash"]:
        raise ValueError("feature-selected prediction belongs to another winner")
    rows = split.holdout_rows
    prediction = base._load_saved_prediction(
        model_config.artifact_path / manifest["prediction"], rows.size
    )
    # This is the only feature-selection function that indexes holdout targets.
    target = data.metadata["target"][rows]
    online_time = data.metadata["time_online"][rows]
    series_position = data.metadata["series_position"][rows]
    score = float(base.tsauc(target, prediction, online_time))
    bootstrap = base.cluster_bootstrap_tsauc(
        target,
        prediction,
        online_time,
        series_position,
        replicates=model_config.bootstrap_replicates,
        seed=model_config.bootstrap_seed + 1,
    )
    result = {
        "feature_selection_version": _workflow_version(model_config),
        "frozen_contract_hash": frozen["contract_hash"],
        "method": frozen["method"],
        "feature_count": frozen["feature_count"],
        "holdout_series": int(split.holdout_series_mask.sum()),
        "holdout_rows": int(rows.size),
        "holdout_tsauc": score,
        "holdout_ordinary_roc_auc_diagnostic": float(
            roc_auc_score(target, prediction)
        ),
        "complete_series_bootstrap": bootstrap,
        "other_feature_candidates_not_scored_on_holdout": True,
        "interpretation": (
            "Approximate submission score for the single method/count frozen "
            "from nested development CV. Do not tune after reading it."
        ),
    }
    base._json_dump(result_path, result)
    low, high = bootstrap["percentile_95_interval"]
    experiment_title = str(getattr(model_config, "experiment_title", "Model 4"))
    report = f"""# {experiment_title} feature-selected holdout result

- Winning method: `{frozen['method']}`
- Selected features: `{frozen['feature_count']}`
- Development mean TSAUC: `{frozen['development_tsauc']['fold_mean']:.6f}`
- Holdout TSAUC: `{score:.6f}`
- Complete-series bootstrap SE: `{bootstrap['standard_error']:.6f}`
- Bootstrap percentile 95% interval: `[{low:.6f}, {high:.6f}]`

Rankings and method/count selection used development folds only. No losing
feature candidate was scored on the holdout.
"""
    (_root(model_config) / "HOLDOUT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    return result


__all__ = [
    "FEATURE_SELECTION_VERSION",
    "SUPPORTED_METHODS",
    "FeatureSelectionConfig",
    "build_feature_rankings",
    "choose_feature_winner",
    "evaluate_feature_selected_holdout",
    "feature_search_table",
    "fit_feature_selected_model",
    "freeze_feature_winner",
    "search_feature_subsets",
]
