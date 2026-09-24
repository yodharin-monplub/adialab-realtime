"""Load frozen development predictions used by the Model 10 study.

The loaders deliberately expose only Model 5 development rows. In particular,
they never load Model 6's saved outer-holdout prediction. The primary source is
the incumbent post-processed Model 6 OOF vector; the pre-global vector can also
be reconstructed for historical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tsauc import tsauc


DEFAULT_METADATA = "models/artifacts/core_features.metadata.npz"
DEFAULT_SPLITS = "models/model5/artifacts/split_assignments.npz"
DEFAULT_TARGET_INDEX = "train/y_train_index.parquet"
DEFAULT_MODEL6_RUN = "models/model6/runs/tsroc0p64_909b8f5c6c"
DEFAULT_MODEL9_STACK = "models/model9/runs/fulldata_819a001db6/pairwise_stack"


@dataclass(frozen=True)
class DevelopmentLayout:
    """Arrays aligned to the Model 5 development-only OOF row order."""

    development_rows: np.ndarray
    target: np.ndarray
    time_online: np.ndarray
    series_position: np.ndarray
    row_fold: np.ndarray
    inner_series_fold: np.ndarray
    outer_series_fold: np.ndarray
    online_lengths: np.ndarray
    tau_index: np.ndarray
    series_ids: np.ndarray

    @property
    def row_count(self) -> int:
        """Return the number of development rows."""

        return int(self.development_rows.size)

    @property
    def series_count(self) -> int:
        """Return the number of complete development series."""

        return int(np.sum(self.inner_series_fold >= 0))


@dataclass(frozen=True)
class SourceValidation:
    """Saved and recomputed TSAUC values for one frozen prediction source."""

    expected_fold_scores: tuple[float, ...]
    observed_fold_scores: tuple[float, ...]
    expected_pooled_score: float
    observed_pooled_score: float
    maximum_absolute_error: float
    tolerance: float


@dataclass(frozen=True)
class FrozenDevelopmentSource:
    """A probability vector plus provenance and score-integrity checks."""

    name: str
    probability: np.ndarray
    summary_path: Path
    validation: SourceValidation
    interpretation: str


def _resolve_under_root(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes project root: {value}") from error
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_vector(path: Path, row_count: int, *, probability: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (row_count,):
        raise ValueError(
            f"development OOF row alignment failed for {path}: "
            f"expected {(row_count,)}, found {values.shape}"
        )
    output = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError(f"prediction contains NaN or infinity: {path}")
    if probability and np.any((output < 0.0) | (output > 1.0)):
        raise ValueError(f"prediction is not a probability vector: {path}")
    return output


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(bounded) - np.log1p(-bounded)


def _logistic(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _validate_scores(
    probability: np.ndarray,
    layout: DevelopmentLayout,
    expected_fold_scores: tuple[float, ...],
    expected_pooled_score: float,
    tolerance: float,
) -> SourceValidation:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("score validation tolerance must be positive and finite")
    fold_ids = tuple(sorted(int(value) for value in np.unique(layout.row_fold)))
    if fold_ids != tuple(range(len(expected_fold_scores))):
        raise ValueError(
            "saved score count does not match Model 5 development folds: "
            f"scores={len(expected_fold_scores)}, folds={fold_ids}"
        )
    observed = tuple(
        float(
            tsauc(
                layout.target[layout.row_fold == fold],
                probability[layout.row_fold == fold],
                layout.time_online[layout.row_fold == fold],
            )
        )
        for fold in fold_ids
    )
    pooled = float(tsauc(layout.target, probability, layout.time_online))
    errors = [
        abs(actual - expected)
        for actual, expected in zip(observed, expected_fold_scores, strict=True)
    ]
    errors.append(abs(pooled - expected_pooled_score))
    maximum_error = float(max(errors, default=0.0))
    if maximum_error > tolerance:
        raise ValueError(
            "frozen source TSAUC validation failed: "
            f"maximum error {maximum_error:.3g} exceeds {tolerance:.3g}"
        )
    return SourceValidation(
        expected_fold_scores=expected_fold_scores,
        observed_fold_scores=observed,
        expected_pooled_score=float(expected_pooled_score),
        observed_pooled_score=pooled,
        maximum_absolute_error=maximum_error,
        tolerance=float(tolerance),
    )


def load_development_layout(
    project_root: str | Path,
    *,
    metadata_path: str | Path = DEFAULT_METADATA,
    split_path: str | Path = DEFAULT_SPLITS,
    target_index_path: str | Path = DEFAULT_TARGET_INDEX,
    folds: int = 5,
) -> DevelopmentLayout:
    """Load Model 5's series-disjoint development rows and fold assignments."""

    root = Path(project_root).resolve()
    metadata_file = _resolve_under_root(root, metadata_path)
    split_file = _resolve_under_root(root, split_path)
    target_index_file = _resolve_under_root(root, target_index_path)
    with np.load(metadata_file, allow_pickle=False) as saved:
        required = {
            "series_position",
            "time_online",
            "online_lengths",
            "series_ids",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"metadata is missing arrays: {sorted(missing)}")
        full_series = saved["series_position"].astype(np.int32, copy=True)
        full_time = saved["time_online"].astype(np.int32, copy=True)
        online_lengths = saved["online_lengths"].astype(np.int32, copy=True)
        series_ids = saved["series_ids"].astype(np.int64, copy=True)
    with np.load(split_file, allow_pickle=False) as saved:
        required = {"inner_series_fold", "outer_series_fold"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"split assignments are missing arrays: {sorted(missing)}")
        inner = saved["inner_series_fold"].astype(np.int8, copy=True)
        outer = saved["outer_series_fold"].astype(np.int8, copy=True)

    if folds < 2:
        raise ValueError("folds must be at least two")
    if not (
        full_series.ndim == full_time.ndim == 1
        and full_series.shape == full_time.shape
    ):
        raise ValueError("row metadata arrays must be aligned one-dimensional vectors")
    if inner.ndim != 1 or outer.shape != inner.shape:
        raise ValueError("series split arrays must be aligned one-dimensional vectors")
    if online_lengths.shape != inner.shape or series_ids.shape != inner.shape:
        raise ValueError("series metadata and split arrays are not aligned")
    if full_series.size and (
        int(full_series.min()) < 0 or int(full_series.max()) >= inner.size
    ):
        raise ValueError("row series positions exceed split assignments")
    if np.any((inner < -1) | (inner >= folds)):
        raise ValueError("inner series fold is outside [-1, folds)")

    development_series = inner >= 0
    if np.any(development_series & (outer == 0)):
        raise AssertionError("sealed outer-fold series entered development rows")
    development_rows = np.flatnonzero(development_series[full_series])
    series = full_series[development_rows]
    online = full_time[development_rows]
    row_fold = inner[series]

    # Query only development IDs from the series-level label index. The full
    # target and sealed holdout tau values in the shared metadata archive are
    # deliberately never loaded by Model 10.
    import pyarrow.dataset as pyarrow_dataset

    development_ids = series_ids[development_series]
    table = pyarrow_dataset.dataset(
        target_index_file, format="parquet"
    ).to_table(
        columns=["id", "tau_index"],
        filter=pyarrow_dataset.field("id").isin(development_ids.tolist()),
    )
    loaded_ids = table["id"].to_numpy(zero_copy_only=False).astype(
        np.int64, copy=False
    )
    loaded_tau = table["tau_index"].to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )
    order = np.argsort(loaded_ids, kind="stable")
    loaded_ids = loaded_ids[order]
    loaded_tau = loaded_tau[order]
    expected_ids = np.sort(development_ids)
    if not np.array_equal(loaded_ids, expected_ids):
        raise ValueError("filtered target index did not return exactly development IDs")
    positions = np.searchsorted(loaded_ids, development_ids)
    tau_index = np.full(inner.size, -99, dtype=np.int32)
    tau_index[development_series] = loaded_tau[positions]
    active_tau = tau_index[development_series]
    active_lengths = online_lengths[development_series]
    if np.any((active_tau < -1) | (active_tau >= active_lengths)):
        raise ValueError("development tau_index is outside its online horizon")
    row_tau = tau_index[series]
    target = ((row_tau >= 0) & (online >= row_tau)).astype(np.uint8)
    expected_folds = set(range(folds))
    if set(np.unique(row_fold).tolist()) != expected_folds:
        raise ValueError("development rows do not contain exactly the requested folds")
    if np.any(series[1:] < series[:-1]):
        raise ValueError("development rows are not grouped in ascending series order")
    same_series = series[1:] == series[:-1]
    if np.any(online[1:][same_series] <= online[:-1][same_series]):
        raise ValueError("time_online is not strictly increasing inside each series")

    return DevelopmentLayout(
        development_rows=development_rows,
        target=target,
        time_online=online,
        series_position=series,
        row_fold=row_fold,
        inner_series_fold=inner,
        outer_series_fold=outer,
        online_lengths=online_lengths,
        tau_index=tau_index,
        series_ids=series_ids,
    )


def _model6_recipe_prediction(
    project_root: Path,
    row_count: int,
    component_paths: Mapping[str, str],
    recipe: Mapping[str, Any],
) -> np.ndarray:
    start = recipe.get("start_component")
    steps = recipe.get("steps")
    if not isinstance(start, str) or start not in component_paths:
        raise ValueError(f"invalid Model 6 start component: {start!r}")
    if not isinstance(steps, list):
        raise ValueError("Model 6 recipe steps must be a list")

    start_path = _resolve_under_root(project_root, component_paths[start])
    current = _safe_logit(_load_vector(start_path, row_count, probability=True))
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Model 6 recipe step {index} is not an object")
        component = step.get("component")
        if not isinstance(component, str) or component not in component_paths:
            raise ValueError(f"invalid Model 6 recipe component: {component!r}")
        alpha = float(step.get("new_component_weight", np.nan))
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"invalid Model 6 recipe weight: {alpha!r}")
        path = _resolve_under_root(project_root, component_paths[component])
        component_margin = _safe_logit(_load_vector(path, row_count, probability=True))
        current *= 1.0 - alpha
        current += alpha * component_margin
    return _logistic(current)


def load_model6_pre_global_postprocess(
    project_root: str | Path,
    *,
    layout: DevelopmentLayout | None = None,
    model6_directory: str | Path = DEFAULT_MODEL6_RUN,
    score_tolerance: float = 1e-6,
) -> FrozenDevelopmentSource:
    """Reconstruct Model 6's frozen global OOF score before final decay."""

    root = Path(project_root).resolve()
    ready_layout = layout if layout is not None else load_development_layout(root)
    model6_path = _resolve_under_root(root, model6_directory)
    summary_path = model6_path / "ensemble/global/summary.json"
    summary = _read_json(summary_path)
    components = summary.get("components")
    held_recipes = summary.get("held_recipes")
    raw_scores = summary.get("nested_raw_tsroc")
    if not isinstance(components, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in components.items()
    ):
        raise ValueError("Model 6 summary has an invalid component map")
    if not isinstance(held_recipes, list) or not isinstance(raw_scores, dict):
        raise ValueError("Model 6 summary is missing raw held-fold recipes or scores")

    expected_fold_scores = tuple(float(value) for value in raw_scores["fold_scores"])
    expected_folds = set(range(len(expected_fold_scores)))
    prediction = np.full(ready_layout.row_count, np.nan, dtype=np.float32)
    seen_folds: set[int] = set()
    for held in held_recipes:
        if not isinstance(held, dict):
            raise ValueError("Model 6 held recipe is not an object")
        fold = int(held.get("held_fold", -1))
        if fold not in expected_folds or fold in seen_folds:
            raise ValueError(f"duplicate or unexpected Model 6 held fold: {fold}")
        recipe = held.get("recipe")
        if not isinstance(recipe, dict):
            raise ValueError(f"Model 6 fold {fold} has no valid recipe")
        all_rows = _model6_recipe_prediction(
            root, ready_layout.row_count, components, recipe
        )
        mask = ready_layout.row_fold == fold
        prediction[mask] = all_rows[mask]
        seen_folds.add(fold)
    if seen_folds != expected_folds or not np.all(np.isfinite(prediction)):
        raise ValueError("Model 6 reconstruction did not fill every development fold")

    validation = _validate_scores(
        prediction,
        ready_layout,
        expected_fold_scores,
        float(raw_scores["pooled_development_oof"]),
        score_tolerance,
    )
    return FrozenDevelopmentSource(
        name="model6_pre_global_postprocess",
        probability=prediction,
        summary_path=summary_path,
        validation=validation,
        interpretation=(
            "Frozen Model 6 nested global OOF reconstructed before its final "
            "causal decayed-maximum transform; Model 6 component selection is reused."
        ),
    )


def load_model6_incumbent_postprocess(
    project_root: str | Path,
    *,
    layout: DevelopmentLayout | None = None,
    model6_directory: str | Path = DEFAULT_MODEL6_RUN,
    score_tolerance: float = 1e-6,
) -> FrozenDevelopmentSource:
    """Load Model 6 after the causal decay used by its frozen deployment."""

    root = Path(project_root).resolve()
    ready_layout = layout if layout is not None else load_development_layout(root)
    model6_path = _resolve_under_root(root, model6_directory)
    summary_path = model6_path / "ensemble/global/summary.json"
    summary = _read_json(summary_path)
    postprocess = summary.get("postprocess")
    prediction_path = summary.get("ensemble_prediction_path")
    if not isinstance(postprocess, dict) or not isinstance(prediction_path, str):
        raise ValueError("Model 6 summary is missing incumbent postprocess output")
    nested_scores = postprocess.get("nested_tsroc")
    if not isinstance(nested_scores, dict):
        raise ValueError("Model 6 summary is missing incumbent nested TSAUC")
    probability = _load_vector(
        _resolve_under_root(root, prediction_path),
        ready_layout.row_count,
        probability=True,
    )
    validation = _validate_scores(
        probability,
        ready_layout,
        tuple(float(value) for value in nested_scores["fold_scores"]),
        float(nested_scores["pooled_development_oof"]),
        score_tolerance,
    )
    transform = str(postprocess.get("deployment_transform", "unknown"))
    return FrozenDevelopmentSource(
        name="model6_incumbent_postprocess",
        probability=probability,
        summary_path=summary_path,
        validation=validation,
        interpretation=(
            "Frozen Model 6 nested global OOF after its incumbent "
            f"{transform} causal transform; Model 6 upstream choices are reused."
        ),
    )


def load_model9_pairwise_stack(
    project_root: str | Path,
    *,
    layout: DevelopmentLayout | None = None,
    model9_directory: str | Path = DEFAULT_MODEL9_STACK,
    score_tolerance: float = 1e-6,
) -> FrozenDevelopmentSource:
    """Load Model 9's exploratory pairwise OOF margin as probabilities."""

    root = Path(project_root).resolve()
    ready_layout = layout if layout is not None else load_development_layout(root)
    model9_path = _resolve_under_root(root, model9_directory)
    summary_path = model9_path / "summary.json"
    summary = _read_json(summary_path)
    saved_scores = summary.get("pairwise_stack")
    prediction_path = summary.get("oof_prediction_path")
    if not isinstance(saved_scores, dict) or not isinstance(prediction_path, str):
        raise ValueError("Model 9 summary is missing pairwise scores or OOF path")
    margin_path = _resolve_under_root(root, prediction_path)
    margin = _load_vector(margin_path, ready_layout.row_count, probability=False)
    probability = _logistic(margin).astype(np.float32)
    expected_fold_scores = tuple(float(value) for value in saved_scores["fold_scores"])
    validation = _validate_scores(
        probability,
        ready_layout,
        expected_fold_scores,
        float(saved_scores["pooled_development_oof"]),
        score_tolerance,
    )
    interpretation = str(
        summary.get(
            "interpretation",
            "Exploratory Model 9 stack; upstream base-model fitting was not fully nested.",
        )
    )
    return FrozenDevelopmentSource(
        name="model9_pairwise_stack",
        probability=probability,
        summary_path=summary_path,
        validation=validation,
        interpretation=interpretation,
    )


def load_frozen_development_sources(
    project_root: str | Path,
    *,
    include_model9: bool = True,
    score_tolerance: float = 1e-6,
) -> tuple[DevelopmentLayout, dict[str, FrozenDevelopmentSource]]:
    """Load the shared layout and all requested frozen Model 10 baselines."""

    root = Path(project_root).resolve()
    layout = load_development_layout(root)
    model6 = load_model6_incumbent_postprocess(
        root, layout=layout, score_tolerance=score_tolerance
    )
    sources = {model6.name: model6}
    if include_model9:
        model9 = load_model9_pairwise_stack(
            root, layout=layout, score_tolerance=score_tolerance
        )
        sources[model9.name] = model9
    return layout, sources


__all__ = [
    "DEFAULT_METADATA",
    "DEFAULT_MODEL6_RUN",
    "DEFAULT_MODEL9_STACK",
    "DEFAULT_SPLITS",
    "DEFAULT_TARGET_INDEX",
    "DevelopmentLayout",
    "FrozenDevelopmentSource",
    "SourceValidation",
    "load_development_layout",
    "load_frozen_development_sources",
    "load_model6_pre_global_postprocess",
    "load_model6_incumbent_postprocess",
    "load_model9_pairwise_stack",
]
