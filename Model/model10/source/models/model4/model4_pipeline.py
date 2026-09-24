"""Resumable detector screening and model training for ``model4.ipynb``.

This module deliberately keeps holdout evaluation separate from every model
selection function. Complete series are split before row sampling, detector
banks are screened only with development out-of-fold predictions, and the
joint-versus-blend decision is frozen before holdout labels are indexed.

The notebook is the user-facing runner. Functions live here so every stage is
testable, restartable, and small enough to inspect independently.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import the repository implementation, not copied detector code. This also
# configures the legacy sibling imports used inside detector modules.
from competition_data import TrainingData  # noqa: E402
from feature_pipeline import (  # noqa: E402
    MODEL4_CORE_DETECTOR_BANKS,
    MODEL4_CANDIDATE_DETECTOR_BANKS,
    ARCHIVE_CANDIDATE_BANKS,
    ARCHIVE_CORE_BANKS,
    CANDIDATE_BANKS,
    CORE_BANKS,
    FeatureCache,
    NOVEL_BANKS,
    SPECIALIST_BANKS,
    bank_slices,
    build_feature_cache,
    feature_names,
    load_feature_cache,
)
from train_cv import (  # noqa: E402
    DEFAULT_MODEL_PARAMETERS,
    HorizontalFeatureMatrix,
    deterministic_sample_mask,
    make_series_folds,
    make_tsauc_evaluator,
    matrix_rows,
    pair_weights_from_reference,
    predict_rows,
    selected_columns,
)
from tsauc import fold_standard_error, tsauc  # noqa: E402


EXPERIMENT_VERSION = "model4-all-43-detectors-v3"
LEGACY_SPECIALIST_BANKS = (
    "d10",
    "d11",
    "d12",
    "d16",
    "d21",
    "d23",
    "d24",
    "d25",
    "d36",
)
LEGACY_SPECIALIST_SIGNATURE = (
    "604fd1c68c9fc5f8a8bc5c5785a2fbb55098dc571636fbd79cf270aaa7ae0fc3"
)
LEGACY_SAFE_SPECIALIST_BANKS = (
    "d10",
    "d11",
    "d12",
    "d21",
    "d23",
    "d24",
    "d25",
    "d36",
)
REPAIRED_SPECIALIST_BANKS = ("d16", "d28", "d29")


@dataclass(frozen=True)
class Model4Config:
    """All choices that can change a fitted model or reported score."""

    project_root: Path = PROJECT_ROOT
    train_directory: str = "train"
    cache_directory: str = "models/artifacts"
    # Store the corrected 43-bank run separately from any legacy experiment.
    artifact_directory: str = "models/model4/all_detector_artifacts"
    outer_folds: int = 5
    holdout_fold: int = 0
    outer_seed: int = 73_051
    development_folds: int = 5
    development_seed: int = 84_127
    train_sample_fraction: float = 0.20
    row_sample_seed: int = 99_173
    detector_roc_threshold: float = 0.50
    screen_num_boost_round: int = 500
    screen_early_stopping_rounds: int = 60
    combined_num_boost_round: int = 1_000
    combined_early_stopping_rounds: int = 100
    screen_model_seed: int = 51_007
    combined_model_seed: int = 61_413
    blend_grid_step: float = 0.05
    prediction_chunk_rows: int = 50_000
    bootstrap_replicates: int = 200
    bootstrap_seed: int = 20_260_904
    cache_build_workers: int = 4
    num_threads: int = max(1, min(8, os.cpu_count() or 1))
    log_every: int = 50

    @property
    def train_path(self) -> Path:
        return self.project_root / self.train_directory

    @property
    def cache_path(self) -> Path:
        return self.project_root / self.cache_directory

    @property
    def artifact_path(self) -> Path:
        return self.project_root / self.artifact_directory

    def validate(self) -> None:
        if self.outer_folds < 2 or not 0 <= self.holdout_fold < self.outer_folds:
            raise ValueError("invalid outer fold configuration")
        # Every reported fold needs three disjoint roles: model fitting,
        # early stopping, and scoring. Two folds would leave no fitting rows.
        if self.development_folds < 3:
            raise ValueError("development_folds must be at least three")
        if not 0.0 < self.train_sample_fraction <= 1.0:
            raise ValueError("train_sample_fraction must be in (0, 1]")
        if not 0.0 <= self.detector_roc_threshold < 1.0:
            raise ValueError("detector_roc_threshold must be in [0, 1)")
        if not 0.0 < self.blend_grid_step <= 1.0:
            raise ValueError("blend_grid_step must be in (0, 1]")
        if self.bootstrap_replicates < 20:
            raise ValueError("use at least 20 complete-series bootstrap replicates")
        if min(self.screen_num_boost_round, self.combined_num_boost_round) < 1:
            raise ValueError("boosting-round limits must be positive")
        if min(
            self.screen_early_stopping_rounds,
            self.combined_early_stopping_rounds,
        ) < 1:
            raise ValueError("early-stopping patience must be positive")
        if self.prediction_chunk_rows < 1 or self.num_threads < 1:
            raise ValueError("prediction_chunk_rows and num_threads must be positive")

    def protocol(self) -> dict[str, Any]:
        """Serializable model protocol, excluding machine-specific root paths."""

        values = asdict(self)
        values.pop("project_root")
        return {"experiment_version": EXPERIMENT_VERSION, **values}


@dataclass
class ExperimentData:
    """Training parquet plus row-aligned caches for every detector source."""

    training: TrainingData
    caches: tuple[Any, ...]
    metadata: dict[str, np.ndarray]
    core_matrix: np.ndarray
    candidate_matrix: np.ndarray
    joint_matrix: Any
    core_banks: tuple[str, ...]
    candidate_banks: tuple[str, ...]
    joint_banks: tuple[str, ...]


@dataclass
class ColumnSubsetMatrix:
    """A zero-copy logical matrix exposing selected columns of one mmap."""

    matrix: Any
    columns: np.ndarray

    def __post_init__(self) -> None:
        self.columns = np.asarray(self.columns, dtype=np.int64)
        if self.columns.ndim != 1 or self.columns.size == 0:
            raise ValueError("column subset must be a non-empty one-dimensional array")
        self.shape = (self.matrix.shape[0], self.columns.size)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, local_columns = item
            row_index = np.asarray(rows).reshape(-1)
            local_index = np.asarray(local_columns).reshape(-1)
            return self.matrix[
                np.ix_(row_index, self.columns[local_index])
            ]
        return np.asarray(self.matrix[item])[..., self.columns]


@dataclass(frozen=True)
class SplitAssignment:
    """Series-level holdout and development-CV assignments expanded to rows."""

    outer_series_fold: np.ndarray
    inner_series_fold: np.ndarray
    row_inner_fold: np.ndarray
    development_series_mask: np.ndarray
    holdout_series_mask: np.ndarray
    development_row_mask: np.ndarray
    holdout_row_mask: np.ndarray
    sampled_row_mask: np.ndarray

    @property
    def development_rows(self) -> np.ndarray:
        return np.flatnonzero(self.development_row_mask)

    @property
    def holdout_rows(self) -> np.ndarray:
        return np.flatnonzero(self.holdout_row_mask)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2) + "\n", encoding="utf-8")


def _write_or_validate_json(path: Path, value: Any) -> None:
    """Write an immutable stage contract or verify an existing identical one."""

    ready = _json_ready(value)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != ready:
            raise ValueError(
                f"{path} belongs to a different experiment configuration; "
                "select a new artifact directory before changing the protocol"
            )
        return
    _json_dump(path, ready)


def _sha_json(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _column_hash(columns: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(columns, dtype=np.int32).tobytes()
    ).hexdigest()


def _safe_name(value: str) -> str:
    return value.replace(".", "p").replace("/", "_")


def _mean_and_se(values: Sequence[float]) -> dict[str, Any]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 2 or not np.all(np.isfinite(scores)):
        raise ValueError("fold scores must contain at least two finite values")
    return {
        "fold_scores": scores.tolist(),
        "fold_mean": float(scores.mean()),
        "fold_standard_deviation": float(scores.std(ddof=1)),
        "fold_standard_error": fold_standard_error(scores),
    }


def _seeded_parameters(config: Model4Config, seed: int) -> dict[str, Any]:
    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    parameters.update(
        {
            "num_threads": config.num_threads,
            "seed": seed,
            "feature_fraction_seed": seed + 100,
            "bagging_seed": seed + 200,
            "data_random_seed": seed + 300,
        }
    )
    return parameters


def make_roc_evaluator():
    """Unweighted ordinary row-level ROC used only for detector screening."""

    def evaluate(prediction: np.ndarray, dataset: lgb.Dataset):
        target = dataset.get_label().astype(np.uint8, copy=False)
        return "ordinary_roc_auc", roc_auc_score(target, prediction), True

    return evaluate


def detector_selection_bound(
    mean_unseen_score: float,
    standard_error: float = 0.0,
    standard_error_multiplier: float = 0.0,
) -> float:
    """Return ``mean - multiplier * SE`` for a detector's unseen folds."""

    values = np.asarray(
        [mean_unseen_score, standard_error, standard_error_multiplier],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        return float("nan")
    if standard_error < 0.0 or standard_error_multiplier < 0.0:
        raise ValueError("standard error and its multiplier must be non-negative")
    return float(mean_unseen_score - standard_error_multiplier * standard_error)


def detector_is_selected(
    mean_unseen_score: float,
    threshold: float = 0.5,
    *,
    standard_error: float = 0.0,
    standard_error_multiplier: float = 0.0,
) -> bool:
    """Apply a strict lower-bound-above-chance detector rule."""

    bound = detector_selection_bound(
        mean_unseen_score,
        standard_error,
        standard_error_multiplier,
    )
    return bool(np.isfinite(bound) and bound > float(threshold))


def _experiment_version(config: Model4Config) -> str:
    """Allow successor experiments to reuse this audited implementation."""

    return str(getattr(config, "experiment_version", EXPERIMENT_VERSION))


def _selection_metric(config: Model4Config) -> str:
    metric = str(getattr(config, "detector_selection_metric", "roc_auc"))
    if metric not in {"roc_auc", "tsauc"}:
        raise ValueError("detector_selection_metric must be 'roc_auc' or 'tsauc'")
    return metric


def _selection_se_multiplier(config: Model4Config) -> float:
    multiplier = float(
        getattr(config, "detector_standard_error_multiplier", 0.0)
    )
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("detector_standard_error_multiplier must be finite and >= 0")
    return multiplier


def _selection_metric_label(config: Model4Config) -> str:
    return (
        "ordinary_row_roc_auc"
        if _selection_metric(config) == "roc_auc"
        else "time_stratified_roc_auc"
    )


def _load_known_legacy_specialist_cache(cache_directory: Path) -> FeatureCache:
    """Open the local pre-audit cache, exposing only verified unaffected banks."""

    manifest_path = cache_directory / "specialist_features.manifest.json"
    matrix_path = cache_directory / "specialist_features.f32.mmap"
    metadata_path = cache_directory / "specialist_features.metadata.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("banks", ())) != LEGACY_SPECIALIST_BANKS:
        raise ValueError("unrecognized stale specialist cache bank contract")
    if manifest.get("feature_signature") != LEGACY_SPECIALIST_SIGNATURE:
        raise ValueError("unrecognized stale specialist cache signature")
    expected_bytes = int(manifest["rows"]) * int(manifest["columns"]) * 4
    if not matrix_path.exists() or matrix_path.stat().st_size != expected_bytes:
        raise ValueError("legacy specialist cache byte size is invalid")
    if not metadata_path.exists():
        raise ValueError("legacy specialist cache metadata is missing")

    slices = bank_slices(LEGACY_SPECIALIST_BANKS)
    stored_names = list(manifest["feature_names"])
    for bank in LEGACY_SAFE_SPECIALIST_BANKS:
        bank_slice = slices[bank]
        if stored_names[bank_slice] != feature_names((bank,)):
            raise ValueError(f"legacy {bank} names differ from current source")
    return FeatureCache(
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        rows=int(manifest["rows"]),
        columns=int(manifest["columns"]),
        banks=LEGACY_SPECIALIST_BANKS,
    )


def load_model4_data(config: Model4Config) -> ExperimentData:
    """Load ``train/`` and all 43 detectors, building only missing caches."""

    config.validate()
    config.artifact_path.mkdir(parents=True, exist_ok=True)
    config.cache_path.mkdir(parents=True, exist_ok=True)
    training = TrainingData(config.train_path)

    # Prefer a current full specialist cache. The user's existing cache is a
    # known pre-audit version: d16 is stale and d28/d29 are absent. Preserve
    # the file, project out d16, and build only the three repaired banks.
    specialist_mode = "current"
    specialist_cache: FeatureCache | None = None
    specialist_manifest = config.cache_path / "specialist_features.manifest.json"
    if specialist_manifest.exists():
        try:
            specialist_cache = load_feature_cache(
                config.cache_path, "specialist_features"
            )
            if specialist_cache.banks != tuple(SPECIALIST_BANKS):
                raise ValueError("specialist cache has an incomplete current contract")
        except ValueError:
            specialist_cache = _load_known_legacy_specialist_cache(config.cache_path)
            specialist_mode = "legacy_safe_projection_plus_repaired_cache"

    cache_specs: list[tuple[str, tuple[str, ...]]] = [
        ("core_features", CORE_BANKS),
        ("candidate_features", CANDIDATE_BANKS),
        ("novel_features", NOVEL_BANKS),
        ("archive_core_features", ARCHIVE_CORE_BANKS),
        ("archive_candidate_features", ARCHIVE_CANDIDATE_BANKS),
    ]
    if specialist_cache is None:
        cache_specs.insert(2, ("specialist_features", SPECIALIST_BANKS))
    elif specialist_mode != "current":
        cache_specs.insert(
            2,
            ("model4_repaired_specialist_features", REPAIRED_SPECIALIST_BANKS),
        )
    missing_specs = [
        (cache_name, banks)
        for cache_name, banks in cache_specs
        if not (config.cache_path / f"{cache_name}.manifest.json").exists()
    ]
    if missing_specs:
        required = 0
        for cache_name, banks in missing_specs:
            expected_bytes = training.online_row_count * len(feature_names(banks)) * 4
            partial_path = config.cache_path / f"{cache_name}.f32.mmap.partial"
            allocated_bytes = (
                partial_path.stat().st_blocks * 512 if partial_path.exists() else 0
            )
            required += max(expected_bytes - allocated_bytes, 0)
        available = shutil.disk_usage(config.cache_path).free
        # Leave room for LightGBM checkpoints, OOF predictions, and filesystem
        # overhead. Partial cache files are intentionally not double-counted.
        reserve = 5 * 1024**3
        if required + reserve > available:
            raise OSError(
                "insufficient free disk for missing all-detector caches: "
                f"need about {required / 1024**3:.1f} GiB plus a 5 GiB reserve, "
                f"but only {available / 1024**3:.1f} GiB is free"
            )

    for cache_name, banks in cache_specs:
        manifest = config.cache_path / f"{cache_name}.manifest.json"
        if not manifest.exists():
            gib = training.online_row_count * len(feature_names(banks)) * 4 / 1024**3
            print(
                f"building missing {cache_name} from train/ "
                f"({len(banks)} banks, {len(feature_names(banks))} columns, "
                f"about {gib:.1f} GiB)",
                flush=True,
            )
            build_feature_cache(
                training,
                cache_directory=config.cache_path,
                name=cache_name,
                banks=banks,
                workers=config.cache_build_workers,
            )

    cache_by_name: dict[str, FeatureCache] = {}
    for cache_name, expected_banks in cache_specs:
        cache = load_feature_cache(config.cache_path, cache_name)
        if tuple(cache.banks) != tuple(expected_banks):
            raise ValueError(
                f"{cache_name} banks differ from the all-detector contract: "
                f"{tuple(cache.banks)}"
            )
        cache_by_name[cache_name] = cache
    if specialist_cache is None:
        specialist_cache = cache_by_name["specialist_features"]

    if specialist_mode == "current":
        ordered_blocks = [
            (cache_by_name["core_features"].open("r"), tuple(CORE_BANKS)),
            (cache_by_name["candidate_features"].open("r"), tuple(CANDIDATE_BANKS)),
            (specialist_cache.open("r"), tuple(SPECIALIST_BANKS)),
            (cache_by_name["novel_features"].open("r"), tuple(NOVEL_BANKS)),
            (
                cache_by_name["archive_core_features"].open("r"),
                tuple(ARCHIVE_CORE_BANKS),
            ),
            (
                cache_by_name["archive_candidate_features"].open("r"),
                tuple(ARCHIVE_CANDIDATE_BANKS),
            ),
        ]
    else:
        legacy_columns, _ = selected_columns(
            specialist_cache.banks, LEGACY_SAFE_SPECIALIST_BANKS
        )
        legacy_view = ColumnSubsetMatrix(
            specialist_cache.open("r"), legacy_columns
        )
        ordered_blocks = [
            (cache_by_name["core_features"].open("r"), tuple(CORE_BANKS)),
            (cache_by_name["candidate_features"].open("r"), tuple(CANDIDATE_BANKS)),
            (legacy_view, tuple(LEGACY_SAFE_SPECIALIST_BANKS)),
            (cache_by_name["novel_features"].open("r"), tuple(NOVEL_BANKS)),
            (
                cache_by_name["model4_repaired_specialist_features"].open("r"),
                tuple(REPAIRED_SPECIALIST_BANKS),
            ),
            (
                cache_by_name["archive_core_features"].open("r"),
                tuple(ARCHIVE_CORE_BANKS),
            ),
            (
                cache_by_name["archive_candidate_features"].open("r"),
                tuple(ARCHIVE_CANDIDATE_BANKS),
            ),
        ]

    joint_matrix = HorizontalFeatureMatrix([matrix for matrix, _ in ordered_blocks])
    joint_banks = tuple(
        bank for _, block_banks in ordered_blocks for bank in block_banks
    )
    core_banks = tuple(MODEL4_CORE_DETECTOR_BANKS)
    candidate_banks = tuple(MODEL4_CANDIDATE_DETECTOR_BANKS)
    if len(set(joint_banks)) != 43 or set(joint_banks) != set(
        core_banks + candidate_banks
    ):
        raise ValueError("all-detector caches do not resolve to exactly 43 banks")
    all_caches = list(cache_by_name.values())
    if specialist_cache not in all_caches:
        all_caches.append(specialist_cache)
    metadata = cache_by_name["core_features"].load_metadata()
    for cache in all_caches:
        candidate_metadata = cache.load_metadata()
        for key in metadata:
            if not np.array_equal(metadata[key], candidate_metadata[key]):
                raise ValueError(
                    f"cache {cache.matrix_path.name} is not row-aligned ({key})"
                )
    if not np.array_equal(metadata["series_ids"], training.series_ids):
        raise ValueError("feature-cache series order differs from train/")
    if not np.array_equal(metadata["online_lengths"], training.online_lengths):
        raise ValueError("feature-cache online lengths differ from train/")
    if not np.array_equal(
        metadata["historical_lengths"], training.historical_lengths
    ):
        raise ValueError("feature-cache historical lengths differ from train/")
    if not np.array_equal(metadata["tau_index"], training.tau_index):
        raise ValueError("feature-cache targets differ from train/")

    cache_signatures = {
        cache_name: json.loads(cache.manifest_path.read_text(encoding="utf-8"))[
            "feature_signature"
        ]
        for cache_name, cache in cache_by_name.items()
    }
    cache_signatures.setdefault(
        "specialist_features",
        json.loads(specialist_cache.manifest_path.read_text(encoding="utf-8"))[
            "feature_signature"
        ],
    )
    cache_contract = {
        "protocol": config.protocol(),
        "protocol_hash": _sha_json(config.protocol()),
        "train_series": len(training),
        "online_rows": training.online_row_count,
        "core_banks": list(core_banks),
        "candidate_banks": list(candidate_banks),
        "core_feature_count": len(feature_names(core_banks)),
        "candidate_feature_count": len(feature_names(candidate_banks)),
        "joint_feature_count": len(feature_names(tuple(joint_banks))),
        "specialist_cache_mode": specialist_mode,
        "cache_manifests": cache_signatures,
        "archived_learned_detector_policy": {
            "d19": "fixed identity basis over causal d3 innovation aspects",
            "d20": "fixed analytic emission margin",
            "d31": "fixed eight-vector DCT basis over historical quantile bins",
            "labels_or_pretrained_artifacts_used": False,
        },
    }
    _write_or_validate_json(config.artifact_path / "experiment_manifest.json", cache_contract)
    return ExperimentData(
        training=training,
        caches=tuple(all_caches),
        metadata=metadata,
        # Physical feature columns follow joint_banks. All three architectures
        # use that same virtual matrix and select their own global columns.
        core_matrix=joint_matrix,
        candidate_matrix=joint_matrix,
        joint_matrix=joint_matrix,
        core_banks=core_banks,
        candidate_banks=candidate_banks,
        joint_banks=tuple(joint_banks),
    )


def assign_series_splits(
    online_lengths: np.ndarray,
    tau_index: np.ndarray,
    *,
    outer_folds: int,
    holdout_fold: int,
    outer_seed: int,
    development_folds: int,
    development_seed: int,
    development_from_outer_folds: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a sealed outer holdout and complete-series development folds.

    By default the development folds are freshly stratified, preserving the
    Model4 protocol. Successor experiments can instead reuse every non-holdout
    outer fold directly; this requires exactly one more outer fold than
    development folds.
    """

    outer = make_series_folds(
        online_lengths, tau_index, n_splits=outer_folds, seed=outer_seed
    ).astype(np.int8)
    inner = np.full(outer.size, -1, dtype=np.int8)
    development_positions = np.flatnonzero(outer != holdout_fold)
    if development_from_outer_folds:
        if outer_folds != development_folds + 1:
            raise ValueError(
                "reusing outer folds requires outer_folds == development_folds + 1"
            )
        development_outer_folds = [
            fold for fold in range(outer_folds) if fold != holdout_fold
        ]
        for inner_fold, outer_fold in enumerate(development_outer_folds):
            inner[outer == outer_fold] = inner_fold
    else:
        inner_subset = make_series_folds(
            np.asarray(online_lengths)[development_positions],
            np.asarray(tau_index)[development_positions],
            n_splits=development_folds,
            seed=development_seed,
        ).astype(np.int8)
        inner[development_positions] = inner_subset
    if np.any(inner[outer == holdout_fold] != -1):
        raise AssertionError("holdout series entered development CV")
    if set(np.unique(inner[development_positions])) != set(range(development_folds)):
        raise AssertionError("development folds are incomplete")
    return outer, inner


def prepare_split(data: ExperimentData, config: Model4Config) -> SplitAssignment:
    """Create, validate, and save all series and row split assignments."""

    outer, inner = assign_series_splits(
        data.metadata["online_lengths"],
        data.metadata["tau_index"],
        outer_folds=config.outer_folds,
        holdout_fold=config.holdout_fold,
        outer_seed=config.outer_seed,
        development_folds=config.development_folds,
        development_seed=config.development_seed,
        development_from_outer_folds=bool(
            getattr(config, "development_from_outer_folds", False)
        ),
    )
    development_series = outer != config.holdout_fold
    holdout_series = ~development_series
    series_position = data.metadata["series_position"]
    row_inner = inner[series_position]
    development_rows = development_series[series_position]
    holdout_rows = holdout_series[series_position]
    sampled = deterministic_sample_mask(
        data.metadata["row_id"],
        data.metadata["time_online"],
        config.train_sample_fraction,
        config.row_sample_seed,
    )
    if np.any(development_rows & holdout_rows):
        raise AssertionError("development and holdout rows overlap")
    if not np.all(development_rows | holdout_rows):
        raise AssertionError("some rows have no split role")

    def role_summary(series_mask: np.ndarray) -> dict[str, int]:
        return {
            "series": int(np.sum(series_mask)),
            "broken_series": int(
                np.sum(data.metadata["tau_index"][series_mask] >= 0)
            ),
            "rows": int(np.sum(data.metadata["online_lengths"][series_mask])),
        }

    contract = {
        "experiment_version": _experiment_version(config),
        "outer": {
            "folds": config.outer_folds,
            "seed": config.outer_seed,
            "sealed_holdout_fold": config.holdout_fold,
            "development": role_summary(development_series),
            "holdout": role_summary(holdout_series),
        },
        "development_cv": {
            "folds": config.development_folds,
            "seed": (
                config.outer_seed
                if bool(getattr(config, "development_from_outer_folds", False))
                else config.development_seed
            ),
            "fold_summary": {
                str(fold): role_summary(inner == fold)
                for fold in range(config.development_folds)
            },
        },
        "row_sampling": {
            "fraction": config.train_sample_fraction,
            "seed": config.row_sample_seed,
            "development_sample_rows": int(np.sum(development_rows & sampled)),
        },
        "leakage_rule": (
            "All detector, feature, blend, round, and architecture decisions use "
            "development folds only. Holdout labels are indexed only by "
            "evaluate_holdout after frozen_contract.json exists."
        ),
    }
    if bool(getattr(config, "development_from_outer_folds", False)):
        if config.outer_folds == 6 and config.development_folds == 5:
            # Preserve the existing Model 5 checkpoint contract byte-for-byte.
            construction = (
                "the five non-holdout folds from one six-way stratified split"
            )
        else:
            construction = (
                f"the {config.development_folds} non-holdout folds from one "
                f"{config.outer_folds}-way stratified split"
            )
        contract["development_cv"]["construction"] = construction
    _write_or_validate_json(config.artifact_path / "split_contract.json", contract)
    assignment_path = config.artifact_path / "split_assignments.npz"
    if assignment_path.exists():
        with np.load(assignment_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["outer_series_fold"], outer) or not np.array_equal(
                saved["inner_series_fold"], inner
            ):
                raise ValueError("saved split assignments differ from current protocol")
    else:
        np.savez_compressed(
            assignment_path,
            outer_series_fold=outer,
            inner_series_fold=inner,
        )
    return SplitAssignment(
        outer_series_fold=outer,
        inner_series_fold=inner,
        row_inner_fold=row_inner,
        development_series_mask=development_series,
        holdout_series_mask=holdout_series,
        development_row_mask=development_rows,
        holdout_row_mask=holdout_rows,
        sampled_row_mask=sampled,
    )


def split_table(data: ExperimentData, split: SplitAssignment) -> pd.DataFrame:
    """Notebook-friendly series/row summary without exposing holdout outcomes."""

    records = []
    for fold in range(int(np.max(split.inner_series_fold)) + 1):
        series = split.inner_series_fold == fold
        records.append(
            {
                "role": f"development_fold_{fold}",
                "series": int(series.sum()),
                "broken_series": int(np.sum(data.metadata["tau_index"][series] >= 0)),
                "rows": int(np.sum(data.metadata["online_lengths"][series])),
            }
        )
    records.append(
        {
            "role": "sealed_holdout",
            "series": int(split.holdout_series_mask.sum()),
            # Break counts are split-balance metadata, not detector/model scores.
            "broken_series": int(
                np.sum(data.metadata["tau_index"][split.holdout_series_mask] >= 0)
            ),
            "rows": int(np.sum(split.holdout_row_mask)),
        }
    )
    return pd.DataFrame.from_records(records)


def development_fold_roles(
    split: SplitAssignment,
    score_fold: int,
    number_of_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return disjoint train/early-stop/score masks for one development fit.

    The next fold in cyclic order is reserved for early stopping. The scored
    fold therefore cannot influence model fitting or boosting-round choice.
    """

    if number_of_folds < 3:
        raise ValueError("three-way CV needs at least three development folds")
    if not 0 <= score_fold < number_of_folds:
        raise ValueError("score_fold is outside the development-fold range")
    stop_fold = (score_fold + 1) % number_of_folds
    score_mask = split.development_row_mask & (
        split.row_inner_fold == score_fold
    )
    stop_mask = split.development_row_mask & (
        split.row_inner_fold == stop_fold
    )
    train_mask = (
        split.development_row_mask
        & (split.row_inner_fold != score_fold)
        & (split.row_inner_fold != stop_fold)
    )
    if not score_mask.any() or not stop_mask.any() or not train_mask.any():
        raise ValueError("one or more three-way CV roles are empty")
    if np.any(train_mask & stop_mask) or np.any(train_mask & score_mask):
        raise AssertionError("three-way CV roles overlap")
    if np.any(stop_mask & score_mask):
        raise AssertionError("early-stop and score roles overlap")
    if not np.array_equal(
        train_mask | stop_mask | score_mask,
        split.development_row_mask,
    ):
        raise AssertionError("three-way CV roles do not cover development rows")
    return train_mask, stop_mask, score_mask, stop_fold


def _family_components(
    data: ExperimentData, family: str
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    if family == "core":
        return data.core_matrix, data.joint_banks, data.core_banks
    if family == "candidate":
        return data.candidate_matrix, data.joint_banks, data.candidate_banks
    raise ValueError("family must be 'core' or 'candidate'")


def _train_early_stopped(
    *,
    matrix: Any,
    metadata: dict[str, np.ndarray],
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    train_reference_mask: np.ndarray,
    validation_reference_mask: np.ndarray,
    columns: np.ndarray,
    names: Sequence[str],
    parameters: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    log_every: int,
    metric: str,
) -> lgb.Booster:
    """Materialize one sampled train/validation pair and release it promptly."""

    x_train = matrix_rows(matrix, train_rows, columns)
    x_valid = matrix_rows(matrix, validation_rows, columns)
    y_train = metadata["target"][train_rows]
    y_valid = metadata["target"][validation_rows]
    if np.unique(y_train).size < 2 or np.unique(y_valid).size < 2:
        raise ValueError("sampled train/validation rows must contain both classes")

    if metric == "tsauc":
        train_weight = pair_weights_from_reference(
            y_train,
            metadata["time_online"][train_rows],
            metadata["target"][train_reference_mask],
            metadata["time_online"][train_reference_mask],
        )
        valid_weight = pair_weights_from_reference(
            y_valid,
            metadata["time_online"][validation_rows],
            metadata["target"][validation_reference_mask],
            metadata["time_online"][validation_reference_mask],
        )
        evaluator = make_tsauc_evaluator(metadata["time_online"][validation_rows])
    elif metric == "roc_auc":
        # The user explicitly selected ordinary row-level ROC as the detector
        # inclusion criterion. Do not add TSAUC pair weights to this screen.
        train_weight = None
        valid_weight = None
        evaluator = make_roc_evaluator()
    else:
        raise ValueError("metric must be 'roc_auc' or 'tsauc'")

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weight,
        feature_name=list(names),
        params=parameters,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        weight=valid_weight,
        feature_name=list(names),
        reference=train_set,
        params=parameters,
        free_raw_data=True,
    )
    train_set.construct()
    valid_set.construct()
    del x_train, x_valid, y_train, y_valid, train_weight, valid_weight
    gc.collect()
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=[f"development_{metric}"],
        feval=evaluator,
        callbacks=[
            lgb.early_stopping(
                early_stopping_rounds, first_metric_only=True, verbose=True
            ),
            lgb.log_evaluation(period=log_every),
        ],
    )
    del train_set, valid_set
    gc.collect()
    return booster


def _load_saved_prediction(path: Path, expected_rows: int) -> np.ndarray:
    prediction = np.load(path, allow_pickle=False)
    if prediction.shape != (expected_rows,) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"invalid saved prediction: {path}")
    return prediction.astype(np.float32, copy=False)


def screen_detector_bank(
    data: ExperimentData,
    split: SplitAssignment,
    family: str,
    bank: str,
    config: Model4Config,
) -> dict[str, Any]:
    """Fit one bank on all unseen development folds and save every checkpoint."""

    matrix, cached_banks, eligible_banks = _family_components(data, family)
    if bank not in eligible_banks:
        raise ValueError(f"{bank} is not a declared {family} detector")
    selection_metric = _selection_metric(config)
    selection_metric_label = _selection_metric_label(config)
    selection_se_multiplier = _selection_se_multiplier(config)
    columns, names = selected_columns(cached_banks, (bank,))
    directory = config.artifact_path / "screen" / family / _safe_name(bank)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_version": _experiment_version(config),
        "stage": "detector_screen",
        "family": family,
        "bank": bank,
        "columns": columns.tolist(),
        "column_hash": _column_hash(columns),
        "feature_count": int(columns.size),
        "feature_names": names,
        "folds": config.development_folds,
        "train_sample_fraction": config.train_sample_fraction,
        "row_sample_seed": config.row_sample_seed,
        "screen_num_boost_round": config.screen_num_boost_round,
        "screen_early_stopping_rounds": config.screen_early_stopping_rounds,
        "screen_model_seed": config.screen_model_seed,
        "selection_metric": selection_metric_label,
        "fold_role_protocol": (
            "score fold is untouched; next cyclic fold is early stopping; "
            "remaining folds are training"
        ),
    }
    if selection_metric != "roc_auc" or selection_se_multiplier != 0.0:
        manifest["selection_standard_error_multiplier"] = (
            selection_se_multiplier
        )
    manifest["configuration_hash"] = _sha_json(manifest)
    _write_or_validate_json(directory / "manifest.json", manifest)

    development_rows = split.development_rows
    local_fold = split.row_inner_fold[development_rows]
    oof = np.full(development_rows.size, np.nan, dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    for fold in range(config.development_folds):
        train_full, stop_full, held_mask, stop_fold = development_fold_roles(
            split, fold, config.development_folds
        )
        held_rows = np.flatnonzero(held_mask)
        train_rows = np.flatnonzero(train_full & split.sampled_row_mask)
        valid_stop_rows = np.flatnonzero(stop_full & split.sampled_row_mask)
        model_path = directory / f"fold_{fold}.txt"
        prediction_path = directory / f"fold_{fold}.prediction.f32.npy"
        result_path = directory / f"fold_{fold}.json"
        if model_path.exists() and prediction_path.exists() and result_path.exists():
            prediction = _load_saved_prediction(prediction_path, held_rows.size)
            record = json.loads(result_path.read_text(encoding="utf-8"))
            print(
                f"screen {family}/{bank} fold {fold}: already complete, "
                f"{selection_metric.upper()}={record[selection_metric]:.6f}",
                flush=True,
            )
        else:
            print(
                f"screen {family}/{bank} fold {fold}: "
                f"{train_rows.size:,} train, {valid_stop_rows.size:,} stop, "
                f"{held_rows.size:,} score rows, {columns.size} features",
                flush=True,
            )
            started = time.perf_counter()
            parameters = _seeded_parameters(
                config,
                config.screen_model_seed + 100 * eligible_banks.index(bank) + fold,
            )
            booster = _train_early_stopped(
                matrix=matrix,
                metadata=data.metadata,
                train_rows=train_rows,
                validation_rows=valid_stop_rows,
                train_reference_mask=train_full,
                validation_reference_mask=stop_full,
                columns=columns,
                names=names,
                parameters=parameters,
                num_boost_round=config.screen_num_boost_round,
                early_stopping_rounds=config.screen_early_stopping_rounds,
                log_every=config.log_every,
                metric=selection_metric,
            )
            booster.save_model(model_path, num_iteration=booster.best_iteration)
            prediction = predict_rows(
                booster,
                matrix,
                held_rows,
                columns,
                config.prediction_chunk_rows,
            )
            np.save(prediction_path, prediction.astype(np.float32))
            target = data.metadata["target"][held_rows]
            record = {
                "fold": fold,
                "early_stop_fold": stop_fold,
                "roc_auc": float(roc_auc_score(target, prediction)),
                "tsauc": float(
                    tsauc(target, prediction, data.metadata["time_online"][held_rows])
                ),
                "best_iteration": int(booster.best_iteration),
                "train_sample_rows": int(train_rows.size),
                "early_stop_sample_rows": int(valid_stop_rows.size),
                "full_validation_rows": int(held_rows.size),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            _json_dump(result_path, record)
            del booster, target
            gc.collect()
        oof[local_fold == fold] = prediction
        fold_records.append(record)

    if not np.all(np.isfinite(oof)):
        raise AssertionError(f"{family}/{bank} OOF predictions are incomplete")
    fold_roc = [float(record["roc_auc"]) for record in fold_records]
    fold_tsauc = [float(record["tsauc"]) for record in fold_records]
    target = data.metadata["target"][development_rows]
    roc_statistics = {
        **_mean_and_se(fold_roc),
        "pooled_development_oof": float(roc_auc_score(target, oof)),
    }
    tsauc_statistics = {
        **_mean_and_se(fold_tsauc),
        "pooled_development_oof": float(
            tsauc(target, oof, data.metadata["time_online"][development_rows])
        ),
    }
    selection_statistics = (
        roc_statistics if selection_metric == "roc_auc" else tsauc_statistics
    )
    selection_bound = detector_selection_bound(
        selection_statistics["fold_mean"],
        selection_statistics["fold_standard_error"],
        selection_se_multiplier,
    )
    summary = {
        "family": family,
        "bank": bank,
        "feature_count": int(columns.size),
        "selection_rule": (
            f"mean unseen-fold {selection_metric_label} - "
            f"{selection_se_multiplier:g} * fold SE > "
            f"{config.detector_roc_threshold:g}"
        ),
        "selected": detector_is_selected(
            selection_statistics["fold_mean"],
            config.detector_roc_threshold,
            standard_error=selection_statistics["fold_standard_error"],
            standard_error_multiplier=selection_se_multiplier,
        ),
        "selection": {
            "metric": selection_metric_label,
            "fold_mean": selection_statistics["fold_mean"],
            "fold_standard_error": selection_statistics["fold_standard_error"],
            "standard_error_multiplier": selection_se_multiplier,
            "lower_bound": selection_bound,
            "threshold": float(config.detector_roc_threshold),
        },
        "folds_above_0p5": int(
            np.sum(
                np.asarray(fold_roc if selection_metric == "roc_auc" else fold_tsauc)
                > 0.5
            )
        ),
        "roc_auc": roc_statistics,
        "tsauc_diagnostic": tsauc_statistics,
        "best_iterations": [int(record["best_iteration"]) for record in fold_records],
        "manifest_hash": manifest["configuration_hash"],
    }
    np.save(directory / "development_oof.f32.npy", oof)
    _json_dump(directory / "summary.json", summary)
    return summary


def screen_family(
    data: ExperimentData,
    split: SplitAssignment,
    family: str,
    config: Model4Config,
) -> dict[str, Any]:
    """Screen every declared bank in one family, sequentially for bounded RAM."""

    _, _, banks = _family_components(data, family)
    results = [
        screen_detector_bank(data, split, family, bank, config) for bank in banks
    ]
    ordered = sorted(
        results,
        key=lambda row: (-row["selection"]["lower_bound"], row["bank"]),
    )
    summary = {
        "experiment_version": _experiment_version(config),
        "family": family,
        "selection_metric": _selection_metric_label(config),
        "selection_threshold": config.detector_roc_threshold,
        "selection_standard_error_multiplier": _selection_se_multiplier(config),
        "selected_banks": [row["bank"] for row in results if row["selected"]],
        "excluded_banks": [row["bank"] for row in results if not row["selected"]],
        "ranking": ordered,
    }
    _json_dump(config.artifact_path / "screen" / family / "summary.json", summary)
    return summary


def screen_table(*summaries: dict[str, Any]) -> pd.DataFrame:
    """Flatten screening output for display in the notebook."""

    records = []
    for summary in summaries:
        for row in summary["ranking"]:
            records.append(
                {
                    "family": row["family"],
                    "bank": row["bank"],
                    "features": row["feature_count"],
                    "selected": row["selected"],
                    "selection_metric": row["selection"]["metric"],
                    "selection_bound": row["selection"]["lower_bound"],
                    "mean_ROC": row["roc_auc"]["fold_mean"],
                    "ROC_SE": row["roc_auc"]["fold_standard_error"],
                    "pooled_ROC": row["roc_auc"]["pooled_development_oof"],
                    "mean_TSAUC": row["tsauc_diagnostic"]["fold_mean"],
                    "TSAUC_SE": row["tsauc_diagnostic"]["fold_standard_error"],
                }
            )
    return pd.DataFrame.from_records(records)


def _save_selection_npz(
    path: Path, columns: np.ndarray, names: Sequence[str], banks: Sequence[str]
) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            same = (
                np.array_equal(saved["columns"], columns)
                and saved["names"].tolist() == list(names)
                and saved["banks"].tolist() == list(banks)
            )
            if not same:
                raise ValueError(f"saved selection differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        columns=np.asarray(columns, dtype=np.int32),
        names=np.asarray(list(names), dtype=np.str_),
        banks=np.asarray(list(banks), dtype=np.str_),
    )


def save_selected_detectors(
    data: ExperimentData,
    core_screen: dict[str, Any],
    candidate_screen: dict[str, Any],
    config: Model4Config,
) -> dict[str, Any]:
    """Freeze detector inclusion before any combined model is evaluated."""

    selected_core = tuple(
        bank for bank in data.core_banks if bank in core_screen["selected_banks"]
    )
    selected_candidate = tuple(
        bank
        for bank in data.candidate_banks
        if bank in candidate_screen["selected_banks"]
    )
    if not selected_core:
        raise RuntimeError(
            "no core detector passed the configured unseen-score rule; inspect the screen "
            "report and explicitly change the threshold before continuing"
        )
    if not selected_candidate:
        raise RuntimeError(
            "no candidate detector passed the configured unseen-score rule; inspect the "
            "screen report and explicitly change the threshold before continuing"
        )

    core_columns, core_names = selected_columns(data.joint_banks, selected_core)
    candidate_columns, candidate_names = selected_columns(
        data.joint_banks, selected_candidate
    )
    joint_banks = selected_core + selected_candidate
    joint_columns, joint_names = selected_columns(data.joint_banks, joint_banks)
    selection = {
        "experiment_version": _experiment_version(config),
        "selection_rule": (
            f"mean of {config.development_folds} unseen-fold "
            f"{_selection_metric_label(config)} - "
            f"{_selection_se_multiplier(config):g} * fold SE > "
            f"{config.detector_roc_threshold:g}"
        ),
        "selected_core_banks": list(selected_core),
        "excluded_core_banks": [
            bank for bank in data.core_banks if bank not in selected_core
        ],
        "selected_candidate_banks": list(selected_candidate),
        "excluded_candidate_banks": [
            bank for bank in data.candidate_banks if bank not in selected_candidate
        ],
        "core_feature_count": int(core_columns.size),
        "candidate_feature_count": int(candidate_columns.size),
        "joint_feature_count": int(joint_columns.size),
        "core_column_hash": _column_hash(core_columns),
        "candidate_column_hash": _column_hash(candidate_columns),
        "joint_column_hash": _column_hash(joint_columns),
    }
    selection["selection_hash"] = _sha_json(selection)
    _write_or_validate_json(
        config.artifact_path / "selected_detectors.json", selection
    )
    _save_selection_npz(
        config.artifact_path / "selected_core_features.npz",
        core_columns,
        core_names,
        selected_core,
    )
    _save_selection_npz(
        config.artifact_path / "selected_candidate_features.npz",
        candidate_columns,
        candidate_names,
        selected_candidate,
    )
    _save_selection_npz(
        config.artifact_path / "selected_joint_features.npz",
        joint_columns,
        joint_names,
        joint_banks,
    )
    return selection


def load_selection_arrays(
    config: Model4Config, family: str
) -> tuple[np.ndarray, list[str], tuple[str, ...]]:
    if family not in {"core", "candidate", "joint"}:
        raise ValueError("family must be core, candidate, or joint")
    path = config.artifact_path / f"selected_{family}_features.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"selection contract not found: {path}; run detector selection first"
        )
    with np.load(path, allow_pickle=False) as saved:
        columns = saved["columns"].astype(np.int32)
        names = saved["names"].tolist()
        banks = tuple(saved["banks"].tolist())
    if columns.ndim != 1 or columns.size == 0 or len(names) != columns.size:
        raise ValueError(f"invalid selection contract: {path}")
    if not banks:
        raise ValueError(f"selection contract has no detector banks: {path}")
    return columns, names, banks


def _cross_validate_columns(
    data: ExperimentData,
    split: SplitAssignment,
    *,
    architecture: str,
    matrix: Any,
    columns: np.ndarray,
    names: Sequence[str],
    config: Model4Config,
) -> dict[str, Any]:
    """Development-fold TSAUC CV for a fixed selected-column contract."""

    directory = config.artifact_path / "combined_cv" / architecture
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_version": _experiment_version(config),
        "stage": "combined_development_cv",
        "architecture": architecture,
        "feature_count": int(columns.size),
        "columns": columns.tolist(),
        "column_hash": _column_hash(columns),
        "feature_names": list(names),
        "folds": config.development_folds,
        "train_sample_fraction": config.train_sample_fraction,
        "row_sample_seed": config.row_sample_seed,
        "num_boost_round": config.combined_num_boost_round,
        "early_stopping_rounds": config.combined_early_stopping_rounds,
        "model_seed": config.combined_model_seed,
        "metric": "tsauc",
        "fold_role_protocol": (
            "score fold is untouched; next cyclic fold is early stopping; "
            "remaining folds are training"
        ),
    }
    manifest["configuration_hash"] = _sha_json(manifest)
    _write_or_validate_json(directory / "manifest.json", manifest)

    development_rows = split.development_rows
    local_fold = split.row_inner_fold[development_rows]
    oof = np.full(development_rows.size, np.nan, dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    architecture_offset = {"core": 0, "candidate": 1_000, "joint": 2_000}[
        architecture
    ]
    for fold in range(config.development_folds):
        train_full, stop_full, held_mask, stop_fold = development_fold_roles(
            split, fold, config.development_folds
        )
        held_rows = np.flatnonzero(held_mask)
        train_rows = np.flatnonzero(train_full & split.sampled_row_mask)
        valid_stop_rows = np.flatnonzero(stop_full & split.sampled_row_mask)
        model_path = directory / f"fold_{fold}.txt"
        prediction_path = directory / f"fold_{fold}.prediction.f32.npy"
        result_path = directory / f"fold_{fold}.json"
        if model_path.exists() and prediction_path.exists() and result_path.exists():
            prediction = _load_saved_prediction(prediction_path, held_rows.size)
            record = json.loads(result_path.read_text(encoding="utf-8"))
            print(
                f"combined {architecture} fold {fold}: already complete, "
                f"TSAUC={record['tsauc']:.6f}",
                flush=True,
            )
        else:
            print(
                f"combined {architecture} fold {fold}: "
                f"{train_rows.size:,} train, {valid_stop_rows.size:,} stop, "
                f"{held_rows.size:,} score rows, {columns.size} features",
                flush=True,
            )
            started = time.perf_counter()
            parameters = _seeded_parameters(
                config,
                config.combined_model_seed + architecture_offset + fold,
            )
            booster = _train_early_stopped(
                matrix=matrix,
                metadata=data.metadata,
                train_rows=train_rows,
                validation_rows=valid_stop_rows,
                train_reference_mask=train_full,
                validation_reference_mask=stop_full,
                columns=columns,
                names=names,
                parameters=parameters,
                num_boost_round=config.combined_num_boost_round,
                early_stopping_rounds=config.combined_early_stopping_rounds,
                log_every=config.log_every,
                metric="tsauc",
            )
            booster.save_model(model_path, num_iteration=booster.best_iteration)
            prediction = predict_rows(
                booster,
                matrix,
                held_rows,
                columns,
                config.prediction_chunk_rows,
            )
            np.save(prediction_path, prediction.astype(np.float32))
            target = data.metadata["target"][held_rows]
            record = {
                "fold": fold,
                "early_stop_fold": stop_fold,
                "tsauc": float(
                    tsauc(target, prediction, data.metadata["time_online"][held_rows])
                ),
                "roc_auc": float(roc_auc_score(target, prediction)),
                "best_iteration": int(booster.best_iteration),
                "train_sample_rows": int(train_rows.size),
                "early_stop_sample_rows": int(valid_stop_rows.size),
                "full_validation_rows": int(held_rows.size),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            _json_dump(result_path, record)
            del booster, target
            gc.collect()
        oof[local_fold == fold] = prediction
        fold_records.append(record)

    if not np.all(np.isfinite(oof)):
        raise AssertionError(f"{architecture} development OOF is incomplete")
    fold_tsauc = [float(record["tsauc"]) for record in fold_records]
    fold_roc = [float(record["roc_auc"]) for record in fold_records]
    target = data.metadata["target"][development_rows]
    summary = {
        "architecture": architecture,
        "feature_count": int(columns.size),
        "tsauc": {
            **_mean_and_se(fold_tsauc),
            "pooled_development_oof": float(
                tsauc(target, oof, data.metadata["time_online"][development_rows])
            ),
        },
        "roc_auc_diagnostic": {
            **_mean_and_se(fold_roc),
            "pooled_development_oof": float(roc_auc_score(target, oof)),
        },
        "best_iterations": [int(record["best_iteration"]) for record in fold_records],
        "fixed_refit_rounds": max(
            1,
            int(
                round(
                    float(
                        np.median(
                            [int(record["best_iteration"]) for record in fold_records]
                        )
                    )
                )
            ),
        ),
        "column_hash": _column_hash(columns),
        "manifest_hash": manifest["configuration_hash"],
    }
    np.save(directory / "development_oof.f32.npy", oof)
    _json_dump(directory / "summary.json", summary)
    return summary


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _logistic(margin: np.ndarray) -> np.ndarray:
    values = np.asarray(margin, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def blend_predictions(
    core_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    core_weight: float,
) -> np.ndarray:
    """Blend on log-odds scale so poorly calibrated components stay comparable."""

    if not 0.0 <= core_weight <= 1.0:
        raise ValueError("core_weight must be in [0, 1]")
    margin = (
        core_weight * _safe_logit(core_prediction)
        + (1.0 - core_weight) * _safe_logit(candidate_prediction)
    )
    return _logistic(margin)


def nested_blend_selection(
    target: np.ndarray,
    time_online: np.ndarray,
    fold: np.ndarray,
    core_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    grid_step: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Choose each held fold's weight using the other four folds only."""

    target = np.asarray(target)
    time_online = np.asarray(time_online)
    fold = np.asarray(fold)
    core_prediction = np.asarray(core_prediction)
    candidate_prediction = np.asarray(candidate_prediction)
    if not (
        target.shape
        == time_online.shape
        == fold.shape
        == core_prediction.shape
        == candidate_prediction.shape
    ):
        raise ValueError("nested blend arrays must have identical one-dimensional shape")
    if target.ndim != 1 or target.size == 0:
        raise ValueError("nested blend arrays must be non-empty and one-dimensional")
    if not (
        np.all(np.isfinite(core_prediction))
        and np.all(np.isfinite(candidate_prediction))
    ):
        raise ValueError("nested blend predictions must be finite")
    if not 0.0 < grid_step <= 1.0:
        raise ValueError("grid_step must be in (0, 1]")
    folds = np.sort(np.unique(fold))
    if np.any(folds < 0):
        raise ValueError("nested blend received holdout rows")
    if folds.size < 2:
        raise ValueError("nested blend needs at least two development folds")
    # Include both pure-model endpoints even when grid_step does not divide 1.
    grid = np.unique(np.r_[np.arange(0.0, 1.0, grid_step), 1.0])
    fold_grid = np.empty((folds.size, grid.size), dtype=np.float64)
    prediction_grid: list[np.ndarray] = []
    for column, weight in enumerate(grid):
        prediction = blend_predictions(core_prediction, candidate_prediction, float(weight))
        prediction_grid.append(prediction)
        for row, held_fold in enumerate(folds):
            held = fold == held_fold
            fold_grid[row, column] = tsauc(
                target[held], prediction[held], time_online[held]
            )

    nested_prediction = np.empty(target.size, dtype=np.float64)
    selected_weights = []
    nested_scores = []
    for held_row, held_fold in enumerate(folds):
        tuning_rows = np.arange(folds.size) != held_row
        mean_score = fold_grid[tuning_rows].mean(axis=0)
        best_value = float(np.max(mean_score))
        tied = np.flatnonzero(np.isclose(mean_score, best_value, atol=1e-12, rtol=0.0))
        best = int(tied[np.argmin(np.abs(grid[tied] - 0.5))])
        selected_weights.append(float(grid[best]))
        held = fold == held_fold
        nested_prediction[held] = prediction_grid[best][held]
        nested_scores.append(float(fold_grid[held_row, best]))

    # Nested fold scores estimate the generalization of weight tuning. Once
    # that estimate exists, the deployable weight is tuned on all development
    # OOF folds. The sealed outer holdout remains untouched.
    full_mean_score = fold_grid.mean(axis=0)
    best_value = float(np.max(full_mean_score))
    tied = np.flatnonzero(
        np.isclose(full_mean_score, best_value, atol=1e-12, rtol=0.0)
    )
    deployment_index = int(tied[np.argmin(np.abs(grid[tied] - 0.5))])
    deployment_weight = float(grid[deployment_index])
    deployment_prediction = blend_predictions(
        core_prediction, candidate_prediction, deployment_weight
    )
    result = {
        "method": "nested_logit_blend",
        "grid_step": grid_step,
        "grid": grid.tolist(),
        "core_weight_selected_without_held_fold": selected_weights,
        "deployment_core_weight": deployment_weight,
        "deployment_weight_selection": (
            "largest mean TSAUC across all development OOF folds; ties choose "
            "the weight closest to 0.5"
        ),
        "deployment_grid_mean_tsauc": full_mean_score.tolist(),
        "nested_tsauc": {
            **_mean_and_se(nested_scores),
            "pooled_development_oof": float(
                tsauc(target, nested_prediction, time_online)
            ),
        },
        "fixed_deployment_weight_tsauc": {
            "pooled_development_oof": float(
                tsauc(target, deployment_prediction, time_online)
            ),
            "fold_scores": [
                float(
                    tsauc(
                        target[fold == held_fold],
                        deployment_prediction[fold == held_fold],
                        time_online[fold == held_fold],
                    )
                )
                for held_fold in folds
            ],
        },
        "fold_grid_tsauc": fold_grid.tolist(),
    }
    return result, nested_prediction.astype(np.float32)


def cross_validate_family_models(
    data: ExperimentData,
    split: SplitAssignment,
    config: Model4Config,
) -> dict[str, Any]:
    """Train selected core/candidate CV models and nested-select their blend."""

    core_columns, core_names, _ = load_selection_arrays(config, "core")
    candidate_columns, candidate_names, _ = load_selection_arrays(
        config, "candidate"
    )
    core = _cross_validate_columns(
        data,
        split,
        architecture="core",
        matrix=data.core_matrix,
        columns=core_columns,
        names=core_names,
        config=config,
    )
    candidate = _cross_validate_columns(
        data,
        split,
        architecture="candidate",
        matrix=data.candidate_matrix,
        columns=candidate_columns,
        names=candidate_names,
        config=config,
    )
    core_oof = np.load(
        config.artifact_path / "combined_cv/core/development_oof.f32.npy",
        allow_pickle=False,
    )
    candidate_oof = np.load(
        config.artifact_path / "combined_cv/candidate/development_oof.f32.npy",
        allow_pickle=False,
    )
    development_rows = split.development_rows
    blend, nested_prediction = nested_blend_selection(
        data.metadata["target"][development_rows],
        data.metadata["time_online"][development_rows],
        split.row_inner_fold[development_rows],
        core_oof,
        candidate_oof,
        grid_step=config.blend_grid_step,
    )
    blend["core_column_hash"] = core["column_hash"]
    blend["candidate_column_hash"] = candidate["column_hash"]
    blend_directory = config.artifact_path / "combined_cv" / "blend"
    blend_directory.mkdir(parents=True, exist_ok=True)
    np.save(blend_directory / "nested_development_oof.f32.npy", nested_prediction)
    _json_dump(blend_directory / "summary.json", blend)
    result = {"core": core, "candidate": candidate, "blend": blend}
    _json_dump(config.artifact_path / "combined_cv/family_models_summary.json", result)
    return result


def cross_validate_joint_model(
    data: ExperimentData,
    split: SplitAssignment,
    config: Model4Config,
) -> dict[str, Any]:
    """Train one LightGBM over all selected core and candidate columns."""

    columns, names, _ = load_selection_arrays(config, "joint")
    return _cross_validate_columns(
        data,
        split,
        architecture="joint",
        matrix=data.joint_matrix,
        columns=columns,
        names=names,
        config=config,
    )


def choose_primary_architecture(
    blend_fold_mean_tsauc: float, joint_fold_mean_tsauc: float
) -> str:
    """Predeclared winner rule; an exact tie retains the two-model blend."""

    return "joint" if joint_fold_mean_tsauc > blend_fold_mean_tsauc else "blend"


def freeze_architecture(
    selection: dict[str, Any],
    family_results: dict[str, Any],
    joint_result: dict[str, Any],
    config: Model4Config,
) -> dict[str, Any]:
    """Freeze every deployment choice before holdout targets are evaluated."""

    blend_score = float(family_results["blend"]["nested_tsauc"]["fold_mean"])
    joint_score = float(joint_result["tsauc"]["fold_mean"])
    primary = choose_primary_architecture(blend_score, joint_score)
    contract = {
        "experiment_version": _experiment_version(config),
        "created_without_holdout_scoring": True,
        "selection_hash": selection["selection_hash"],
        "selected_core_banks": selection["selected_core_banks"],
        "selected_candidate_banks": selection["selected_candidate_banks"],
        "architecture_rule": (
            "Choose larger five-fold development mean TSAUC; retain blend on tie."
        ),
        "primary_architecture": primary,
        "development_scores": {
            "nested_blend_fold_mean_tsauc": blend_score,
            "joint_fold_mean_tsauc": joint_score,
        },
        "deployment_core_weight": float(
            family_results["blend"]["deployment_core_weight"]
        ),
        "fixed_rounds": {
            "core": int(family_results["core"]["fixed_refit_rounds"]),
            "candidate": int(family_results["candidate"]["fixed_refit_rounds"]),
            "joint": int(joint_result["fixed_refit_rounds"]),
        },
        "column_hashes": {
            "core": family_results["core"]["column_hash"],
            "candidate": family_results["candidate"]["column_hash"],
            "joint": joint_result["column_hash"],
        },
        "holdout_policy": (
            "Fit with fixed rounds on sampled development rows, score the frozen "
            "primary architecture once, and never use secondary holdout predictions "
            "to revise the winner."
        ),
    }
    contract["contract_hash"] = _sha_json(contract)
    _write_or_validate_json(config.artifact_path / "frozen_contract.json", contract)
    return contract


def _validated_frozen_contract(
    config: Model4Config, supplied: dict[str, Any]
) -> dict[str, Any]:
    """Return the on-disk contract only when the caller passed that exact file."""

    path = config.artifact_path / "frozen_contract.json"
    if not path.exists():
        raise RuntimeError("freeze the architecture before final fitting")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if _json_ready(supplied) != saved:
        raise ValueError("the supplied frozen architecture differs from disk")
    unhashed = dict(saved)
    claimed_hash = unhashed.pop("contract_hash", None)
    if claimed_hash != _sha_json(unhashed):
        raise ValueError("frozen architecture contract hash is invalid")
    if saved.get("primary_architecture") not in {"blend", "joint"}:
        raise ValueError("frozen primary architecture must be blend or joint")
    return saved


def architecture_table(
    family_results: dict[str, Any], joint_result: dict[str, Any]
) -> pd.DataFrame:
    """Compact notebook comparison using development folds only."""

    return pd.DataFrame.from_records(
        [
            {
                "architecture": "core",
                "mean_TSAUC": family_results["core"]["tsauc"]["fold_mean"],
                "TSAUC_SE": family_results["core"]["tsauc"][
                    "fold_standard_error"
                ],
                "pooled_TSAUC": family_results["core"]["tsauc"][
                    "pooled_development_oof"
                ],
            },
            {
                "architecture": "candidate",
                "mean_TSAUC": family_results["candidate"]["tsauc"]["fold_mean"],
                "TSAUC_SE": family_results["candidate"]["tsauc"][
                    "fold_standard_error"
                ],
                "pooled_TSAUC": family_results["candidate"]["tsauc"][
                    "pooled_development_oof"
                ],
            },
            {
                "architecture": "nested_blend",
                "mean_TSAUC": family_results["blend"]["nested_tsauc"]["fold_mean"],
                "TSAUC_SE": family_results["blend"]["nested_tsauc"][
                    "fold_standard_error"
                ],
                "pooled_TSAUC": family_results["blend"]["nested_tsauc"][
                    "pooled_development_oof"
                ],
            },
            {
                "architecture": "joint",
                "mean_TSAUC": joint_result["tsauc"]["fold_mean"],
                "TSAUC_SE": joint_result["tsauc"]["fold_standard_error"],
                "pooled_TSAUC": joint_result["tsauc"][
                    "pooled_development_oof"
                ],
            },
        ]
    )


def _fit_fixed_model(
    *,
    data: ExperimentData,
    split: SplitAssignment,
    matrix: Any,
    columns: np.ndarray,
    names: Sequence[str],
    architecture: str,
    rounds: int,
    seed: int,
    config: Model4Config,
) -> Path:
    """Fit with fixed rounds on development rows; never use holdout early stopping."""

    directory = config.artifact_path / "final" / architecture
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.txt"
    manifest = {
        "experiment_version": _experiment_version(config),
        "stage": "fixed_development_refit",
        "architecture": architecture,
        "rounds": int(rounds),
        "seed": int(seed),
        "feature_count": int(columns.size),
        "column_hash": _column_hash(columns),
        "train_sample_fraction": config.train_sample_fraction,
        "row_sample_seed": config.row_sample_seed,
        "holdout_used_for_training_or_early_stopping": False,
    }
    manifest["configuration_hash"] = _sha_json(manifest)
    _write_or_validate_json(directory / "manifest.json", manifest)
    if model_path.exists():
        print(f"final {architecture}: already fitted", flush=True)
        return model_path

    train_full = split.development_row_mask
    train_rows = np.flatnonzero(train_full & split.sampled_row_mask)
    print(
        f"final {architecture}: fitting {train_rows.size:,} sampled development "
        f"rows, {columns.size:,} features, {rounds} fixed rounds",
        flush=True,
    )
    x_train = matrix_rows(matrix, train_rows, columns)
    y_train = data.metadata["target"][train_rows]
    weights = pair_weights_from_reference(
        y_train,
        data.metadata["time_online"][train_rows],
        data.metadata["target"][train_full],
        data.metadata["time_online"][train_full],
    )
    parameters = _seeded_parameters(config, seed)
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        feature_name=list(names),
        params=parameters,
        free_raw_data=True,
    )
    train_set.construct()
    del x_train, y_train, weights
    gc.collect()
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=int(rounds),
        callbacks=[lgb.log_evaluation(period=config.log_every)],
    )
    booster.save_model(model_path, num_iteration=int(rounds))
    del booster, train_set
    gc.collect()
    return model_path


def _predict_fixed_model(
    model_path: Path,
    matrix: Any,
    rows: np.ndarray,
    columns: np.ndarray,
    output_path: Path,
    config: Model4Config,
) -> np.ndarray:
    if output_path.exists():
        return _load_saved_prediction(output_path, rows.size)
    booster = lgb.Booster(model_file=str(model_path))
    prediction = predict_rows(
        booster, matrix, rows, columns, config.prediction_chunk_rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, prediction.astype(np.float32))
    del booster
    gc.collect()
    return prediction


def fit_final_models(
    data: ExperimentData,
    split: SplitAssignment,
    frozen: dict[str, Any],
    config: Model4Config,
) -> dict[str, Any]:
    """Fit both requested architectures and create predictions without scoring."""

    frozen = _validated_frozen_contract(config, frozen)
    core_columns, core_names, _ = load_selection_arrays(config, "core")
    candidate_columns, candidate_names, _ = load_selection_arrays(
        config, "candidate"
    )
    joint_columns, joint_names, _ = load_selection_arrays(config, "joint")
    specifications = {
        "core": (data.core_matrix, core_columns, core_names, 0),
        "candidate": (data.candidate_matrix, candidate_columns, candidate_names, 1_000),
        "joint": (data.joint_matrix, joint_columns, joint_names, 2_000),
    }
    holdout_rows = split.holdout_rows
    predictions: dict[str, np.ndarray] = {}
    model_paths: dict[str, str] = {}
    for architecture, (matrix, columns, names, offset) in specifications.items():
        model_path = _fit_fixed_model(
            data=data,
            split=split,
            matrix=matrix,
            columns=columns,
            names=names,
            architecture=architecture,
            rounds=int(frozen["fixed_rounds"][architecture]),
            seed=config.combined_model_seed + 10_000 + offset,
            config=config,
        )
        prediction_path = (
            config.artifact_path
            / "final"
            / architecture
            / "holdout_prediction.f32.npy"
        )
        predictions[architecture] = _predict_fixed_model(
            model_path,
            matrix,
            holdout_rows,
            columns,
            prediction_path,
            config,
        )
        model_paths[architecture] = str(model_path.relative_to(config.artifact_path))

    predictions["blend"] = blend_predictions(
        predictions["core"],
        predictions["candidate"],
        float(frozen["deployment_core_weight"]),
    ).astype(np.float32)
    blend_path = config.artifact_path / "final/blend/holdout_prediction.f32.npy"
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    if blend_path.exists():
        saved_blend = _load_saved_prediction(blend_path, holdout_rows.size)
        if not np.array_equal(saved_blend, predictions["blend"]):
            raise ValueError("saved blend prediction differs from current components")
    else:
        np.save(blend_path, predictions["blend"])

    prediction_manifest = {
        "experiment_version": _experiment_version(config),
        "frozen_contract_hash": frozen["contract_hash"],
        "holdout_rows": int(holdout_rows.size),
        "models": model_paths,
        "predictions": {
            "core": "final/core/holdout_prediction.f32.npy",
            "candidate": "final/candidate/holdout_prediction.f32.npy",
            "joint": "final/joint/holdout_prediction.f32.npy",
            "blend": "final/blend/holdout_prediction.f32.npy",
        },
        "holdout_labels_read": False,
    }
    _write_or_validate_json(
        config.artifact_path / "final/prediction_manifest.json",
        prediction_manifest,
    )
    return prediction_manifest


def cluster_bootstrap_tsauc(
    target: np.ndarray,
    prediction: np.ndarray,
    time_online: np.ndarray,
    series_position: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap complete series, preserving within-series time dependence."""

    target = np.asarray(target)
    prediction = np.asarray(prediction)
    time_online = np.asarray(time_online)
    series_position = np.asarray(series_position)
    if not (
        target.shape
        == prediction.shape
        == time_online.shape
        == series_position.shape
    ):
        raise ValueError("bootstrap arrays must have identical shapes")
    if target.ndim != 1 or target.size == 0:
        raise ValueError("bootstrap arrays must be non-empty and one-dimensional")
    if replicates < 2:
        raise ValueError("bootstrap requires at least two replicates")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("bootstrap prediction contains NaN or infinity")
    unique_series = np.unique(series_position)
    groups = [np.flatnonzero(series_position == series) for series in unique_series]
    rng = np.random.default_rng(seed)
    scores = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in sampled])
        scores[replicate] = tsauc(
            target[rows], prediction[rows], time_online[rows]
        )
        if (replicate + 1) % 25 == 0:
            print(
                f"complete-series bootstrap {replicate + 1}/{replicates}",
                flush=True,
            )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "series_clusters": int(unique_series.size),
        "standard_error": float(scores.std(ddof=1)),
        "percentile_95_interval": np.quantile(scores, (0.025, 0.975)).tolist(),
        "bootstrap_scores": scores.tolist(),
    }


def evaluate_holdout(
    data: ExperimentData,
    split: SplitAssignment,
    frozen: dict[str, Any],
    config: Model4Config,
) -> dict[str, Any]:
    """Open the sealed holdout once and score only the frozen primary model."""

    frozen = _validated_frozen_contract(config, frozen)
    result_path = config.artifact_path / "holdout_results.json"
    if result_path.exists():
        saved_result = json.loads(result_path.read_text(encoding="utf-8"))
        if saved_result.get("frozen_contract_hash") != frozen["contract_hash"]:
            raise ValueError("saved holdout result belongs to another contract")
        print("holdout was already evaluated; returning the saved result", flush=True)
        return saved_result
    prediction_manifest_path = config.artifact_path / "final/prediction_manifest.json"
    if not prediction_manifest_path.exists():
        raise RuntimeError("fit final models and predictions before holdout evaluation")
    prediction_manifest = json.loads(prediction_manifest_path.read_text())
    if prediction_manifest["frozen_contract_hash"] != frozen["contract_hash"]:
        raise ValueError("final predictions belong to another frozen contract")

    primary = frozen["primary_architecture"]
    relative_path = prediction_manifest["predictions"][primary]
    prediction = _load_saved_prediction(
        config.artifact_path / relative_path,
        split.holdout_rows.size,
    )
    rows = split.holdout_rows
    # This is the only experiment function that indexes held-out targets.
    target = data.metadata["target"][rows]
    online_time = data.metadata["time_online"][rows]
    series_position = data.metadata["series_position"][rows]
    score = float(tsauc(target, prediction, online_time))
    bootstrap = cluster_bootstrap_tsauc(
        target,
        prediction,
        online_time,
        series_position,
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    result = {
        "experiment_version": _experiment_version(config),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frozen_contract_hash": frozen["contract_hash"],
        "primary_architecture": primary,
        "holdout_series": int(split.holdout_series_mask.sum()),
        "holdout_rows": int(rows.size),
        "holdout_tsauc": score,
        "holdout_ordinary_roc_auc_diagnostic": float(
            roc_auc_score(target, prediction)
        ),
        "complete_series_bootstrap": bootstrap,
        "secondary_holdout_scores_intentionally_not_computed": True,
        "interpretation": (
            "This is the approximate submission score for the architecture "
            "selected before holdout access. Do not change the model in response "
            "to this result and continue calling the same holdout estimate honest."
        ),
    }
    _json_dump(result_path, result)
    experiment_title = str(getattr(config, "experiment_title", "Model 4"))
    report = f"""# {experiment_title} sealed-holdout result

- Frozen primary architecture: `{primary}`
- Holdout TSAUC: `{score:.6f}`
- Complete-series bootstrap SE: `{bootstrap['standard_error']:.6f}`
- Bootstrap percentile 95% interval: `[{bootstrap['percentile_95_interval'][0]:.6f}, {bootstrap['percentile_95_interval'][1]:.6f}]`
- Ordinary row ROC AUC (diagnostic only): `{result['holdout_ordinary_roc_auc_diagnostic']:.6f}`
- Held-out series/rows: `{result['holdout_series']}` / `{result['holdout_rows']}`

The joint and two-model architectures were compared on development OOF TSAUC
before this holdout was opened. Secondary holdout predictions were not scored,
so the frozen winner cannot be changed using holdout labels.
"""
    (config.artifact_path / "HOLDOUT_REPORT.md").write_text(report, encoding="utf-8")
    return result
