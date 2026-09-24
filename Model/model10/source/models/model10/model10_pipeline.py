"""Leakage-controlled Model 10 causal post-processing experiment.

This module never materializes or scores sealed-series labels and never opens
the reduced test files. It scores frozen development OOF predictions and nests transform selection inside
each original OOF fold. The resulting number is conditional evidence about the
post-processing rule, not a new unbiased estimate of the upstream model.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from models.model10.causal_transforms import TransformSpec, apply_transform_grid
from models.model10.config import Model10Config
from models.model10.selection import (
    expand_series_assignments,
    make_nested_series_subfolds,
    paired_standard_error,
    select_nested_candidates,
    summarize_foldwise_gain,
)
from models.model10.sources import (
    DevelopmentLayout,
    FrozenDevelopmentSource,
    load_development_layout,
    load_model6_incumbent_postprocess,
    load_model9_pairwise_stack,
)
from tsauc import fold_standard_error, tsauc, tsauc_details


def _json_ready(value: Any) -> Any:
    """Convert NumPy and dataclass values into stable JSON primitives."""

    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    """Write a JSON artifact atomically so interrupted runs remain resumable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically save a small compressed NumPy checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    """Atomically save one NumPy array."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    os.replace(temporary, path)


def _array_signature(*arrays: np.ndarray) -> str:
    """Hash aligned array contents, shapes, and dtypes for checkpoint binding."""

    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _score_rows(
    layout: DevelopmentLayout,
    prediction: np.ndarray,
    rows: np.ndarray,
) -> float:
    """Return exact TSAUC on one precomputed row subset."""

    return float(
        tsauc(
            layout.target[rows],
            prediction[rows],
            layout.time_online[rows],
        )
    )


def _fold_and_cell_rows(
    row_fold: np.ndarray,
    row_subfold: np.ndarray,
    folds: int,
    subfolds: int,
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[np.ndarray, ...], ...]]:
    """Precompute disjoint row indexes for repeated exact metric calls."""

    fold_rows = tuple(np.flatnonzero(row_fold == fold) for fold in range(folds))
    cell_rows = tuple(
        tuple(
            np.flatnonzero((row_fold == fold) & (row_subfold == cell))
            for cell in range(subfolds)
        )
        for fold in range(folds)
    )
    covered = sum(rows.size for group in cell_rows for rows in group)
    if covered != row_fold.size:
        raise AssertionError("nested score cells do not cover development rows")
    return fold_rows, cell_rows


def _load_or_create_subfolds(
    config: Model10Config,
    layout: DevelopmentLayout,
    layout_signature: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic within-OOF series splits and persist their contract."""

    path = config.artifact_path / "nested_series_subfolds.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            series_subfold = saved["series_subfold"].astype(np.int8, copy=True)
            saved_signature = str(saved["layout_signature"].item())
        if series_subfold.shape != layout.inner_series_fold.shape:
            raise ValueError(f"stale nested subfold artifact: {path}")
        if saved_signature != layout_signature:
            raise ValueError(f"nested subfold source signature changed: {path}")
    else:
        series_subfold = make_nested_series_subfolds(
            layout.inner_series_fold,
            layout.online_lengths,
            layout.tau_index,
            seed=config.seed,
            n_subfolds=config.calibration_subfolds,
        )
        _atomic_npz(
            path,
            series_subfold=series_subfold,
            layout_signature=np.asarray(layout_signature),
        )

    row_subfold_full = expand_series_assignments(
        layout.series_position, series_subfold
    )
    if np.any(row_subfold_full < 0):
        raise AssertionError("sealed series entered Model 10 development rows")
    return series_subfold, row_subfold_full.astype(np.int8, copy=False)


def _load_grid_checkpoint(
    path: Path,
    names: tuple[str, ...],
    folds: int,
    subfolds: int,
    layout_signature: str,
    source_signature: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a compatible score checkpoint or return empty score matrices."""

    shape = (len(names), folds, subfolds)
    outer_shape = (len(names), folds)
    if not path.exists():
        return np.full(shape, np.nan), np.full(outer_shape, np.nan)
    with np.load(path, allow_pickle=False) as saved:
        prior_names = tuple(str(value) for value in saved["candidate_names"])
        cells = saved["cell_scores"].astype(np.float64, copy=True)
        outer = saved["outer_fold_scores"].astype(np.float64, copy=True)
        prior_layout_signature = str(saved["layout_signature"].item())
        prior_source_signature = str(saved["source_signature"].item())
    if prior_names != names or cells.shape != shape or outer.shape != outer_shape:
        raise ValueError(f"incompatible Model 10 checkpoint: {path}")
    if (
        prior_layout_signature != layout_signature
        or prior_source_signature != source_signature
    ):
        raise ValueError(f"Model 10 checkpoint input signature changed: {path}")
    return cells, outer


def _save_grid_checkpoint(
    path: Path,
    names: tuple[str, ...],
    cell_scores: np.ndarray,
    outer_scores: np.ndarray,
    layout_signature: str,
    source_signature: str,
) -> None:
    _atomic_npz(
        path,
        candidate_names=np.asarray(names),
        cell_scores=cell_scores,
        outer_fold_scores=outer_scores,
        layout_signature=np.asarray(layout_signature),
        source_signature=np.asarray(source_signature),
    )


def _score_transform_grid(
    config: Model10Config,
    layout: DevelopmentLayout,
    source: FrozenDevelopmentSource,
    row_subfold: np.ndarray,
    source_directory: Path,
    layout_signature: str,
    source_signature: str,
) -> tuple[tuple[TransformSpec, ...], np.ndarray, np.ndarray]:
    """Score or resume every predeclared transform on held folds and cells."""

    specs = config.transform_specs()
    names = tuple(spec.name for spec in specs)
    checkpoint = source_directory / "grid_scores.npz"
    cell_scores, outer_scores = _load_grid_checkpoint(
        checkpoint,
        names,
        config.folds,
        config.calibration_subfolds,
        layout_signature,
        source_signature,
    )
    complete = (
        np.all(np.isfinite(cell_scores), axis=(1, 2))
        & np.all(np.isfinite(outer_scores), axis=1)
    )
    if np.all(complete):
        print(f"{source.name}: resuming {len(specs)}/{len(specs)} grid scores")
        return specs, cell_scores, outer_scores
    fold_rows, cell_rows = _fold_and_cell_rows(
        layout.row_fold,
        row_subfold,
        config.folds,
        config.calibration_subfolds,
    )
    total = len(specs)
    started = time.monotonic()
    completed_at_start = int(
        np.sum(
            np.all(np.isfinite(cell_scores), axis=(1, 2))
            & np.all(np.isfinite(outer_scores), axis=1)
        )
    )
    if completed_at_start:
        print(f"{source.name}: resuming {completed_at_start}/{total} grid scores")

    for index, (spec, prediction) in enumerate(
        apply_transform_grid(
            source.probability,
            layout.time_online,
            layout.series_position,
            specs,
        )
    ):
        already_done = bool(
            np.all(np.isfinite(cell_scores[index]))
            and np.all(np.isfinite(outer_scores[index]))
        )
        if already_done:
            continue
        for fold in range(config.folds):
            outer_scores[index, fold] = _score_rows(
                layout, prediction, fold_rows[fold]
            )
            for cell in range(config.calibration_subfolds):
                cell_scores[index, fold, cell] = _score_rows(
                    layout, prediction, cell_rows[fold][cell]
                )
        _save_grid_checkpoint(
            checkpoint,
            names,
            cell_scores,
            outer_scores,
            layout_signature,
            source_signature,
        )
        elapsed = time.monotonic() - started
        mean = float(outer_scores[index].mean())
        print(
            f"{source.name} {index + 1:03d}/{total}: "
            f"{spec.name} mean={mean:.9f} elapsed={elapsed / 60.0:.2f} min",
            flush=True,
        )

    if not np.all(np.isfinite(cell_scores)) or not np.all(np.isfinite(outer_scores)):
        raise AssertionError("Model 10 transform grid is incomplete")
    return specs, cell_scores, outer_scores


def _candidate_diagnostics(
    specs: tuple[TransformSpec, ...],
    cell_scores: np.ndarray,
    outer_scores: np.ndarray,
    cell_pair_weights: np.ndarray,
    config: Model10Config,
) -> list[dict[str, Any]]:
    """Summarize fixed candidates; these rankings are exploratory diagnostics."""

    raw_cells = cell_scores[0]
    raw_outer = outer_scores[0]
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        cell_gain_matrix = cell_scores[index] - raw_cells
        fold_cell_gains = np.asarray(
            [
                np.average(cell_gain_matrix[fold], weights=cell_pair_weights[fold])
                for fold in range(config.folds)
            ]
        )
        fold_gains = outer_scores[index] - raw_outer
        cell_se = paired_standard_error(fold_cell_gains)
        fold_se = paired_standard_error(fold_gains)
        rows.append(
            {
                "candidate_index": index,
                "name": spec.name,
                "family": spec.family,
                "spec": spec.to_dict(),
                "outer_fold_scores": outer_scores[index],
                "outer_fold_mean": float(outer_scores[index].mean()),
                "outer_fold_standard_error": fold_standard_error(
                    outer_scores[index]
                ),
                "fold_gains": fold_gains,
                "mean_fold_gain": float(fold_gains.mean()),
                "paired_fold_gain_standard_error": fold_se,
                "robust_fold_gain": float(
                    fold_gains.mean() - config.uncertainty_penalty * fold_se
                ),
                "fold_wins": int(np.sum(fold_gains > 0.0)),
                "pair_weighted_cell_gain_by_fold": fold_cell_gains,
                "mean_cell_gain": float(fold_cell_gains.mean()),
                "paired_cell_gain_standard_error": cell_se,
                "robust_cell_gain": float(
                    fold_cell_gains.mean() - config.uncertainty_penalty * cell_se
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -row["robust_cell_gain"],
            -row["mean_cell_gain"],
            row["candidate_index"],
        ),
    )


def _assemble_nested_prediction(
    layout: DevelopmentLayout,
    source: FrozenDevelopmentSource,
    specs: tuple[TransformSpec, ...],
    row_subfold: np.ndarray,
    selected_candidate: np.ndarray,
) -> np.ndarray:
    """Re-run only the selected candidates and fill their untouched held cells."""

    row_choice = selected_candidate[layout.row_fold, row_subfold]
    selected_indexes = set(int(value) for value in np.unique(row_choice))
    selected_specs = tuple(
        spec for index, spec in enumerate(specs) if index in selected_indexes
    )
    name_to_index = {spec.name: index for index, spec in enumerate(specs)}
    nested = np.full(layout.row_count, np.nan, dtype=np.float32)
    for spec, prediction in apply_transform_grid(
        source.probability,
        layout.time_online,
        layout.series_position,
        selected_specs,
    ):
        candidate_index = name_to_index[spec.name]
        rows = row_choice == candidate_index
        nested[rows] = prediction[rows]
    if not np.all(np.isfinite(nested)):
        raise AssertionError("nested transform assembly left unscored rows")
    return nested


def _choose_deployment_candidate(
    diagnostics: list[dict[str, Any]],
    config: Model10Config,
) -> dict[str, Any]:
    """Choose a single all-development candidate under the frozen safety gate."""

    raw = next(row for row in diagnostics if row["name"] == "raw")
    eligible = [
        row
        for row in diagnostics
        if row["name"] != "raw"
        and row["mean_cell_gain"] >= config.minimum_mean_gain
        and row["robust_cell_gain"] >= config.minimum_robust_gain
        and row["mean_fold_gain"] >= config.minimum_mean_gain
        and row["robust_fold_gain"] >= config.minimum_robust_gain
        and row["fold_wins"] >= config.minimum_outer_fold_wins
    ]
    return eligible[0] if eligible else raw


def _score_nested_result(
    config: Model10Config,
    layout: DevelopmentLayout,
    source: FrozenDevelopmentSource,
    specs: tuple[TransformSpec, ...],
    cell_scores: np.ndarray,
    outer_scores: np.ndarray,
    row_subfold: np.ndarray,
    source_directory: Path,
) -> dict[str, Any]:
    """Select on four cells, score the fifth, and apply the promotion contract."""

    selection = select_nested_candidates(
        cell_scores,
        raw_index=0,
        uncertainty_penalty=config.uncertainty_penalty,
        minimum_mean_gain=config.minimum_mean_gain,
        minimum_robust_gain=config.minimum_robust_gain,
    )
    fold_index, cell_index = np.indices(selection.selected_candidate.shape)
    selected_cell_scores = cell_scores[
        selection.selected_candidate, fold_index, cell_index
    ]
    raw_cell_scores = cell_scores[0]
    cell_pair_weights = np.empty(raw_cell_scores.shape, dtype=np.int64)
    for fold in range(config.folds):
        for cell in range(config.calibration_subfolds):
            rows = np.flatnonzero(
                (layout.row_fold == fold) & (row_subfold == cell)
            )
            cell_pair_weights[fold, cell] = tsauc_details(
                layout.target[rows],
                np.zeros(rows.size, dtype=np.float32),
                layout.time_online[rows],
            ).total_pair_weight
    if np.any(cell_pair_weights <= 0):
        raise ValueError("every nested held cell must contain TSAUC pairs")
    raw_fold_cell_means = np.asarray(
        [
            np.average(raw_cell_scores[fold], weights=cell_pair_weights[fold])
            for fold in range(config.folds)
        ]
    )
    selected_fold_cell_means = np.asarray(
        [
            np.average(
                selected_cell_scores[fold], weights=cell_pair_weights[fold]
            )
            for fold in range(config.folds)
        ]
    )
    held_cell_paired = summarize_foldwise_gain(
        raw_fold_cell_means,
        selected_fold_cell_means,
        uncertainty_penalty=config.uncertainty_penalty,
        minimum_mean_gain=config.minimum_mean_gain,
        minimum_robust_gain=config.minimum_robust_gain,
        minimum_wins=config.minimum_outer_fold_wins,
    )
    held_cell_gains = selected_cell_scores - raw_cell_scores
    nested = _assemble_nested_prediction(
        layout,
        source,
        specs,
        row_subfold,
        selection.selected_candidate,
    )
    raw_margin = next(
        prediction
        for spec, prediction in apply_transform_grid(
            source.probability,
            layout.time_online,
            layout.series_position,
            (specs[0],),
        )
        if spec.name == "raw"
    )
    raw_fold_scores = np.asarray(
        [
            _score_rows(layout, raw_margin, np.flatnonzero(layout.row_fold == fold))
            for fold in range(config.folds)
        ]
    )
    nested_fold_scores = np.asarray(
        [
            _score_rows(layout, nested, np.flatnonzero(layout.row_fold == fold))
            for fold in range(config.folds)
        ]
    )
    assembled_paired = summarize_foldwise_gain(
        raw_fold_scores,
        nested_fold_scores,
        uncertainty_penalty=config.uncertainty_penalty,
        minimum_mean_gain=config.minimum_mean_gain,
        minimum_robust_gain=config.minimum_robust_gain,
        minimum_wins=config.minimum_outer_fold_wins,
    )
    diagnostics = _candidate_diagnostics(
        specs, cell_scores, outer_scores, cell_pair_weights, config
    )
    deployment = _choose_deployment_candidate(diagnostics, config)
    counts = np.bincount(
        selection.selected_candidate.reshape(-1), minlength=len(specs)
    )
    selected_counts = {
        specs[index].name: int(count)
        for index, count in enumerate(counts)
        if count > 0
    }
    stage_one_passed = bool(
        held_cell_paired.promotion_passed and deployment["name"] != "raw"
    )
    report = {
        "source": source.name,
        "source_summary_path": source.summary_path.relative_to(
            config.project_root
        ).as_posix(),
        "source_interpretation": source.interpretation,
        "validation_scope": (
            "Incremental transform selection is nested within frozen original OOF "
            "folds. Upstream model/feature choices reused development data, so this "
            "is conditional evidence and not a fresh leaderboard estimate."
        ),
        "sealed_outer_holdout_labels_used": False,
        "sealed_outer_holdout_rows_materialized": False,
        "rows": layout.row_count,
        "series": layout.series_count,
        "source_integrity": source.validation,
        "raw": {
            "fold_scores": raw_fold_scores,
            "fold_mean": float(raw_fold_scores.mean()),
            "fold_standard_error": fold_standard_error(raw_fold_scores),
            "pooled_tsauc": float(
                tsauc(layout.target, raw_margin, layout.time_online)
            ),
        },
        "nested_selection": {
            "selected_candidate_by_fold_and_held_cell": selection.selected_candidate,
            "selected_name_by_fold_and_held_cell": [
                [specs[int(index)].name for index in row]
                for row in selection.selected_candidate
            ],
            "selection_counts": selected_counts,
            "training_mean_gain": selection.training_mean_gain,
            "training_standard_error": selection.training_standard_error,
            "training_robust_gain": selection.training_robust_gain,
            "held_cell_raw_scores": raw_cell_scores,
            "held_cell_selected_scores": selected_cell_scores,
            "held_cell_gains": held_cell_gains,
            "held_cell_pair_weights": cell_pair_weights,
            "held_cell_wins": int(np.sum(held_cell_gains > 0.0)),
            "held_cell_unclustered_standard_error": paired_standard_error(
                held_cell_gains.reshape(-1)
            ),
            "raw_fold_cell_means": raw_fold_cell_means,
            "selected_fold_cell_means": selected_fold_cell_means,
            "held_cell_raw_mean": float(raw_fold_cell_means.mean()),
            "held_cell_selected_mean": float(selected_fold_cell_means.mean()),
            "equal_cell_raw_mean_diagnostic": float(raw_cell_scores.mean()),
            "equal_cell_selected_mean_diagnostic": float(
                selected_cell_scores.mean()
            ),
            "paired_gain": held_cell_paired,
            "assembled_crossfit_diagnostic": {
                "reason_diagnostic_only": (
                    "Different held cells may select different transforms; scoring "
                    "their union compares heterogeneous margins across cells. The "
                    "held-cell paired result above is the primary selector evidence."
                ),
                "fold_scores": nested_fold_scores,
                "fold_mean": float(nested_fold_scores.mean()),
                "fold_standard_error": fold_standard_error(nested_fold_scores),
                "pooled_tsauc": float(
                    tsauc(layout.target, nested, layout.time_online)
                ),
                "paired_gain": assembled_paired,
            },
        },
        "best_fixed_candidate_diagnostic": diagnostics[0],
        "top_fixed_candidate_diagnostics": diagnostics[:20],
        "all_development_deployment_candidate": deployment,
        "stage_one_promotion_gate_passed": stage_one_passed,
        "bootstrap_status": (
            "not run; a paired complete-series bootstrap is required before "
            "deployment if the stage-one gate passes"
        ),
        "deployment_promoted": False,
        "recommendation": (
            "retain the frozen upstream deployment unchanged; do not add a Model "
            "10 transform without both the nested gate and a confirmatory "
            "complete-series bootstrap"
        ),
    }
    _atomic_npy(source_directory / "nested_selected.margin.f32.npy", nested)
    _atomic_json(source_directory / "summary.json", report)
    return report


def _write_results_markdown(
    config: Model10Config,
    reports: Iterable[dict[str, Any]],
    *,
    filename: str = "RESULTS.md",
) -> None:
    """Write a compact human-readable result next to machine-readable JSON."""

    lines = [
        "# Model 10 Results",
        "",
        "No sealed outer-holdout labels were materialized or used, and no test "
        "labels were opened. Scores below estimate "
        "only the incremental causal post-processing selector conditional on the "
        "already-frozen upstream OOF predictions.",
        "",
        "| Source | Raw full-fold TSAUC ± SE | Held-cell selector gain ± clustered SE | Fold wins | Stage 1 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    ready_reports = list(reports)
    for report in ready_reports:
        raw = report["raw"]
        nested = report["nested_selection"]
        gain = nested["paired_gain"]
        lines.append(
            f"| `{report['source']}` | {raw['fold_mean']:.6f} ± "
            f"{raw['fold_standard_error']:.6f} | {gain.mean_gain:+.6f} ± "
            f"{gain.paired_standard_error:.6f} | {gain.wins}/5 | "
            f"{'pass' if report['stage_one_promotion_gate_passed'] else 'fail'} |"
        )
    for report in ready_reports:
        best = report["best_fixed_candidate_diagnostic"]
        lines.extend(
            [
                "",
                f"- `{report['source']}` best fixed diagnostic: `{best['name']}` "
                f"({best['mean_fold_gain']:+.6f} mean fold gain; selected on these "
                "same folds, so it is not confirmatory).",
                f"- Deployment recommendation: {report['recommendation']}",
                f"- Heterogeneous cross-fit diagnostic gain: "
                f"{report['nested_selection']['assembled_crossfit_diagnostic']['paired_gain'].mean_gain:+.6f} "
                "(not the primary selector estimate).",
            ]
        )
    lines.extend(
        [
            "",
            "A global strictly increasing calibration is intentionally absent: it "
            "cannot change TSAUC because ranking occurs separately at each online time.",
            "",
        ]
    )
    (config.artifact_path / filename).write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_postprocess_study(
    config: Model10Config,
    *,
    include_model9: bool = True,
) -> dict[str, Any]:
    """Run both frozen-source post-process studies and return their summary."""

    config.validate()
    config.artifact_path.mkdir(parents=True, exist_ok=True)
    layout = load_development_layout(config.project_root, folds=config.folds)
    layout_signature = _array_signature(
        layout.development_rows,
        layout.target,
        layout.time_online,
        layout.series_position,
        layout.row_fold,
        layout.inner_series_fold,
        layout.outer_series_fold,
        layout.online_lengths,
        layout.tau_index,
        layout.series_ids,
    )
    _, row_subfold = _load_or_create_subfolds(
        config, layout, layout_signature
    )
    manifest = {
        "protocol": config.protocol(),
        "development_rows": layout.row_count,
        "development_series": layout.series_count,
        "development_layout_sha256": layout_signature,
        "sealed_outer_holdout_labels_used": False,
        "sealed_outer_holdout_rows_materialized": False,
    }
    manifest_path = config.artifact_path / "experiment_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = _json_ready(manifest)
        migrated = dict(prior)
        if migrated.pop("sealed_outer_holdout_read", None) is False:
            migrated["sealed_outer_holdout_labels_used"] = False
            migrated["sealed_outer_holdout_rows_materialized"] = False
        if migrated != expected_manifest:
            raise ValueError(
                f"{manifest_path} belongs to a different Model 10 protocol"
            )
        if prior != expected_manifest:
            _atomic_json(manifest_path, expected_manifest)
    else:
        _atomic_json(manifest_path, manifest)

    sources = [
        load_model6_incumbent_postprocess(
            config.project_root,
            layout=layout,
            model6_directory=config.source_model6_directory,
        )
    ]
    if include_model9:
        sources.append(
            load_model9_pairwise_stack(
                config.project_root,
                layout=layout,
                model9_directory=(
                    Path(config.source_model9_directory) / "pairwise_stack"
                ),
            )
        )

    reports = []
    for source in sources:
        directory = config.artifact_path / source.name
        directory.mkdir(parents=True, exist_ok=True)
        source_signature = _array_signature(source.probability)
        source_contract = {
            "source": source.name,
            "source_summary_path": source.summary_path.relative_to(
                config.project_root
            ).as_posix(),
            "source_prediction_sha256": source_signature,
            "development_layout_sha256": layout_signature,
        }
        source_contract_path = directory / "input_contract.json"
        if source_contract_path.exists():
            prior_contract = json.loads(
                source_contract_path.read_text(encoding="utf-8")
            )
            if prior_contract != source_contract:
                raise ValueError(
                    f"Model 10 source input changed: {source_contract_path}"
                )
        else:
            _atomic_json(source_contract_path, source_contract)
        specs, cells, outer = _score_transform_grid(
            config,
            layout,
            source,
            row_subfold,
            directory,
            layout_signature,
            source_signature,
        )
        report = _score_nested_result(
            config,
            layout,
            source,
            specs,
            cells,
            outer,
            row_subfold,
            directory,
        )
        reports.append(report)

    summary = {
        "experiment": "Model 10 causal post-processing",
        "protocol": config.protocol(),
        "conditional_validation_only": True,
        "sealed_outer_holdout_labels_used": False,
        "sealed_outer_holdout_rows_materialized": False,
        "reports": reports,
    }
    summary_filename = "summary.json" if include_model9 else "summary_model6_only.json"
    results_filename = "RESULTS.md" if include_model9 else "RESULTS_model6_only.md"
    _atomic_json(config.artifact_path / summary_filename, summary)
    _write_results_markdown(config, reports, filename=results_filename)
    return summary


def study_status(config: Model10Config) -> dict[str, Any]:
    """Return paths and completion state without loading prediction artifacts."""

    summary = config.artifact_path / "summary.json"
    return {
        "artifact_directory": config.artifact_path,
        "complete": summary.is_file(),
        "model6_complete": (
            config.artifact_path
            / "model6_incumbent_postprocess/summary.json"
        ).is_file(),
        "model9_complete": (
            config.artifact_path / "model9_pairwise_stack/summary.json"
        ).is_file(),
        "summary_path": summary,
        "results_path": config.artifact_path / "RESULTS.md",
        "model6_only_summary_path": (
            config.artifact_path / "summary_model6_only.json"
        ),
        "candidate_count": len(config.transform_specs()),
    }


__all__ = ["run_postprocess_study", "study_status"]
