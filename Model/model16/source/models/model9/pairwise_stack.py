"""Cross-fitted, TSAUC-aligned linear stack over existing raw OOF predictions.

This experiment is intentionally independent from Model 9's feature-model
scaling pipeline.  It consumes only development out-of-fold predictions:

* the 31 detector-bank probabilities selected by Model 5; and
* the three feature models retained by the timeout-safe Model 6 submission.

Model 5 selected those banks on the same five development folds. Consequently,
the results are reused-development evidence even before the level-two nesting
caveat below; they are not a fresh leakage-free validation score.

For each scored development fold, positive and negative observations are
paired at the same ``time_online`` using only the other four folds.  Pair
strata are sampled in proportion to ``n_positive * n_negative``, exactly the
weight used by TSAUC.  A fixed, strongly regularized logistic model learns on
mirrored logit differences.  Fixing ``C`` in the immutable protocol avoids a
second, easy-to-miss use of the scored folds for hyperparameter selection.

The already-opened Model 6 outer holdout is never loaded.  Nested Model 6
ensembles are not loaded either: every stack column is a primitive raw OOF
vector.  This remains an exploratory level-two estimate, rather than a fully
nested base-model estimate, because a primitive OOF value for training fold
``f`` can come from a base voter trained on the meta-held fold ``h``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsauc import fold_standard_error, tsauc


PAIRWISE_STACK_VERSION = "model9-pairwise-primitive-oof-v1"
DEFAULT_MODEL6_RUN = "models/model6/runs/tsroc0p64_909b8f5c6c"
MODEL5_ARTIFACTS = "models/model5/artifacts"

# These are exactly the feature models retained by Submission 6.  A stack
# using them adds only a dot product at inference; no extra feature learner is
# required in the deployed bundle.
DEPLOYED_FEATURE_MODELS = (
    ("core", "xgboost__fisher_256__xgb_depth3"),
    ("candidate", "extra_trees__gain_512__extra_sparse_columns"),
    ("candidate", "random_forest__fisher_512__rf_shallow"),
)


@dataclass(frozen=True)
class PairwiseStackConfig:
    """Immutable protocol matching the successful exploratory diagnostic."""

    project_root: Path
    model6_directory: str = DEFAULT_MODEL6_RUN
    detector_bank_count: int = 31
    feature_model_count: int = 3
    folds: int = 5
    pairs_per_fit: int = 200_000
    regularization_c: float = 0.001
    logit_epsilon: float = 1e-6
    minimum_pair_scale: float = 1e-3
    solver: str = "lbfgs"
    maximum_iterations: int = 120
    tolerance: float = 1e-6
    fold_pair_seed: int = 9_101
    deployment_pair_seed: int = 9_601
    prediction_chunk_rows: int = 250_000
    uncertainty_penalty: float = 0.5
    artifact_directory: str = ""

    @property
    def model6_path(self) -> Path:
        return self.project_root / self.model6_directory

    @property
    def model5_path(self) -> Path:
        return self.project_root / MODEL5_ARTIFACTS

    @property
    def artifact_path(self) -> Path:
        return self.project_root / self.artifact_directory

    def protocol(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("project_root")
        return {"experiment_version": PAIRWISE_STACK_VERSION, **values}

    def identity_protocol(self) -> dict[str, Any]:
        values = self.protocol()
        values.pop("artifact_directory")
        return values

    def validate(self) -> None:
        if self.detector_bank_count != 31 or self.feature_model_count != 3:
            raise ValueError("the primitive stack contract is fixed at 31 + 3 inputs")
        if self.folds != 5:
            raise ValueError("the primitive stack must align with five Model 6 folds")
        if self.pairs_per_fit < 1:
            raise ValueError("pairs_per_fit must be positive")
        if self.regularization_c != 0.001:
            raise ValueError("C=0.001 is frozen before the confirmatory run")
        if not 0.0 < self.logit_epsilon < 0.5:
            raise ValueError("logit_epsilon must lie in (0, 0.5)")
        if self.minimum_pair_scale <= 0.0:
            raise ValueError("minimum_pair_scale must be positive")
        if self.solver != "lbfgs":
            raise ValueError("the solver is frozen as lbfgs")
        if min(self.maximum_iterations, self.prediction_chunk_rows) < 1:
            raise ValueError("iteration and chunk limits must be positive")
        if self.tolerance <= 0.0 or self.uncertainty_penalty < 0.0:
            raise ValueError("tolerance and uncertainty penalty are invalid")


@dataclass
class StackData:
    """Row-aligned development labels, folds, and primitive logit matrix."""

    target: np.ndarray
    time_online: np.ndarray
    row_fold: np.ndarray
    series_position: np.ndarray
    feature_names: list[str]
    source_paths: list[Path]
    source_hashes: list[str]
    logits: np.ndarray
    development_rows: np.ndarray
    development_series: int
    model5_selection_hash: str
    model6_protocol_hash: str


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


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _write_or_validate(path: Path, value: Any) -> None:
    ready = _json_ready(value)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != ready:
            raise ValueError(
                f"{path} belongs to a different pairwise-stack protocol"
            )
        return
    _atomic_json(path, ready)


def make_config(
    project_root: str | Path, model6_directory: str = DEFAULT_MODEL6_RUN
) -> PairwiseStackConfig:
    """Create the configuration-hashed output directory for this protocol."""

    root = Path(project_root).resolve()
    model6 = Path(model6_directory)
    if model6.is_absolute():
        try:
            model6 = model6.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError("the Model 6 run must be inside the project") from error
    provisional = PairwiseStackConfig(
        project_root=root,
        model6_directory=model6.as_posix(),
    )
    label = _hash_json(provisional.identity_protocol())[:10]
    config = PairwiseStackConfig(
        **{
            **asdict(provisional),
            "project_root": root,
            "artifact_directory": f"models/model9/runs/pairwise34_{label}",
        }
    )
    config.validate()
    return config


def config_for_model9_context(context: Any) -> PairwiseStackConfig:
    """Place pairwise checkpoints below an existing Model 9 run directory.

    The shared ``Model9Config`` intentionally does not duplicate this module's
    experimental constants.  They remain frozen in ``PairwiseStackConfig`` and
    are included in this stage's own protocol hash, while its artifacts live at
    ``context.config.artifact_path / 'pairwise_stack'``.
    """

    shared = context.config
    root = Path(shared.project_root).resolve()
    source = Path(shared.source_model6_directory)
    artifact = Path(shared.artifact_path).resolve() / "pairwise_stack"
    try:
        artifact_relative = artifact.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("the Model 9 artifact path must be inside the project") from error
    config = PairwiseStackConfig(
        project_root=root,
        model6_directory=source.as_posix(),
        artifact_directory=artifact_relative,
    )
    config.validate()
    return config


def run_for_model9_context(context: Any) -> dict[str, Any]:
    """Run this self-contained stage inside ``context.config.artifact_path``."""

    return run(config_for_model9_context(context))


def run_pairwise_stack(context: Any) -> dict[str, Any]:
    """Public Model 9 runner/notebook alias for the pairwise stage."""

    return run_for_model9_context(context)


def _development_only_path(path: Path, project_root: Path) -> str:
    """Reject accidental outer-holdout inputs before any array is opened."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"stack input is outside the project: {path}") from error
    lowered = tuple(part.lower() for part in relative.parts)
    if any("holdout" in part for part in lowered):
        raise ValueError(f"outer-holdout path is prohibited: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return relative.as_posix()


def _source_contract(
    config: PairwiseStackConfig,
) -> tuple[list[str], list[Path], dict[str, Any], dict[str, Any]]:
    selection_path = config.model5_path / "selected_detectors.json"
    model6_manifest_path = config.model6_path / "experiment_manifest.json"
    _development_only_path(selection_path, config.project_root)
    _development_only_path(model6_manifest_path, config.project_root)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model6_manifest = json.loads(model6_manifest_path.read_text(encoding="utf-8"))
    if model6_manifest.get("source_selection_hash") != selection.get("selection_hash"):
        raise ValueError("Model 5 selection and Model 6 run do not match")

    feature_names: list[str] = []
    source_paths: list[Path] = []
    for family, key in (
        ("core", "selected_core_banks"),
        ("candidate", "selected_candidate_banks"),
    ):
        banks = list(selection[key])
        for bank in banks:
            feature_names.append(f"{family}:bank:{bank}")
            source_paths.append(
                config.model5_path
                / "screen"
                / family
                / bank
                / "development_oof.f32.npy"
            )
    if len(source_paths) != config.detector_bank_count:
        raise ValueError(
            f"expected {config.detector_bank_count} selected banks, got "
            f"{len(source_paths)}"
        )

    for family, component in DEPLOYED_FEATURE_MODELS:
        feature_names.append(f"{family}:feature:{component}")
        source_paths.append(
            config.model6_path
            / "final_cv"
            / family
            / component
            / "raw_oof.f32.npy"
        )
    if len(source_paths) != config.detector_bank_count + config.feature_model_count:
        raise AssertionError("the primitive input contract has the wrong width")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("primitive stack feature names are not unique")
    for path in source_paths:
        _development_only_path(path, config.project_root)
    return feature_names, source_paths, selection, model6_manifest


def load_stack_data(config: PairwiseStackConfig) -> StackData:
    """Load and hash the 34 primitive development OOF vectors."""

    config.validate()
    feature_names, source_paths, selection, model6_manifest = _source_contract(config)
    metadata_path = config.project_root / "models/artifacts/core_features.metadata.npz"
    split_path = config.model5_path / "split_assignments.npz"
    for path in (metadata_path, split_path):
        _development_only_path(path, config.project_root)

    with np.load(metadata_path, allow_pickle=False) as saved:
        full_series = saved["series_position"].astype(np.int32, copy=False)
        full_target = saved["target"].astype(np.uint8, copy=False)
        full_time = saved["time_online"].astype(np.int16, copy=False)
    with np.load(split_path, allow_pickle=False) as saved:
        inner_series_fold = saved["inner_series_fold"].astype(np.int8, copy=False)
        outer_series_fold = saved["outer_series_fold"].astype(np.int8, copy=False)

    if full_series.ndim != 1 or inner_series_fold.ndim != 1:
        raise ValueError("row or series split metadata is malformed")
    if int(full_series.max(initial=-1)) >= inner_series_fold.size:
        raise ValueError("row series positions exceed split assignments")
    development_series_mask = inner_series_fold >= 0
    if np.any(development_series_mask & (outer_series_fold == 0)):
        raise AssertionError("sealed outer series entered the development population")
    development_rows = np.flatnonzero(development_series_mask[full_series])
    target = full_target[development_rows].copy()
    time_online = full_time[development_rows].copy()
    series_position = full_series[development_rows].copy()
    row_fold = inner_series_fold[series_position].copy()
    if set(np.unique(row_fold).tolist()) != set(range(config.folds)):
        raise ValueError("development OOF rows do not contain exactly five folds")

    row_count = development_rows.size
    logits = np.empty((row_count, len(source_paths)), dtype=np.float32)
    source_hashes: list[str] = []
    for column, path in enumerate(source_paths):
        probability = np.load(path, mmap_mode="r", allow_pickle=False)
        if probability.shape != (row_count,):
            raise ValueError(f"OOF row alignment failed for {path}: {probability.shape}")
        if not np.all(np.isfinite(probability)):
            raise ValueError(f"OOF prediction contains non-finite values: {path}")
        bounded = np.clip(
            np.asarray(probability, dtype=np.float32),
            config.logit_epsilon,
            1.0 - config.logit_epsilon,
        )
        logits[:, column] = np.log(bounded) - np.log1p(-bounded)
        source_hashes.append(_sha256_file(path))
        del probability, bounded
        print(
            f"pairwise stack input {column + 1:02d}/{len(source_paths)}: "
            f"{feature_names[column]}",
            flush=True,
        )

    return StackData(
        target=target,
        time_online=time_online,
        row_fold=row_fold,
        series_position=series_position,
        feature_names=feature_names,
        source_paths=source_paths,
        source_hashes=source_hashes,
        logits=logits,
        development_rows=development_rows,
        development_series=int(development_series_mask.sum()),
        model5_selection_hash=str(selection["selection_hash"]),
        model6_protocol_hash=str(model6_manifest["protocol_hash"]),
    )


def sample_time_matched_pairs(
    target: np.ndarray,
    time_online: np.ndarray,
    eligible_mask: np.ndarray,
    pair_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample positive-negative row pairs with official TSAUC stratum weights."""

    y = np.asarray(target, dtype=np.uint8)
    step = np.asarray(time_online, dtype=np.int64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if not (y.shape == step.shape == eligible.shape) or y.ndim != 1:
        raise ValueError("pair-sampling arrays must be aligned one-dimensional vectors")
    if pair_count < 1 or not eligible.any():
        raise ValueError("pair sampling requires eligible rows and a positive count")

    positions = np.flatnonzero(eligible)
    sorted_positions = positions[
        np.argsort(step[positions], kind="stable")
    ]
    sorted_time = step[sorted_positions]
    starts = np.flatnonzero(np.r_[True, sorted_time[1:] != sorted_time[:-1]])
    ends = np.r_[starts[1:], sorted_positions.size]

    positive_groups: list[np.ndarray] = []
    negative_groups: list[np.ndarray] = []
    group_times: list[int] = []
    pair_weights: list[int] = []
    for left, right in zip(starts, ends):
        group = sorted_positions[left:right]
        positive = group[y[group] == 1]
        negative = group[y[group] == 0]
        if positive.size and negative.size:
            positive_groups.append(positive)
            negative_groups.append(negative)
            group_times.append(int(step[group[0]]))
            pair_weights.append(int(positive.size * negative.size))
    if not pair_weights:
        raise ValueError("eligible rows contain no scoreable TSAUC strata")

    probabilities = np.asarray(pair_weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    chosen_group = rng.choice(
        len(pair_weights), size=pair_count, replace=True, p=probabilities
    )
    positive_rows = np.empty(pair_count, dtype=np.int64)
    negative_rows = np.empty(pair_count, dtype=np.int64)
    for group_index in np.unique(chosen_group):
        slots = np.flatnonzero(chosen_group == group_index)
        positive_rows[slots] = rng.choice(
            positive_groups[group_index], size=slots.size, replace=True
        )
        negative_rows[slots] = rng.choice(
            negative_groups[group_index], size=slots.size, replace=True
        )

    if not np.all(step[positive_rows] == step[negative_rows]):
        raise AssertionError("a sampled pair crossed time_online strata")
    if not np.all(y[positive_rows] == 1) or not np.all(y[negative_rows] == 0):
        raise AssertionError("a sampled pair has reversed or non-binary labels")
    detail = {
        "seed": int(seed),
        "pairs": int(pair_count),
        "eligible_rows": int(positions.size),
        "scoreable_time_strata": int(len(pair_weights)),
        "minimum_time_online": int(min(group_times)),
        "maximum_time_online": int(max(group_times)),
        "total_available_pair_weight": int(sum(pair_weights)),
        "sampling_rule": (
            "time strata proportional to n_positive*n_negative; one positive "
            "and one negative sampled uniformly with replacement within stratum"
        ),
    }
    return positive_rows, negative_rows, detail


def fit_pairwise_model(
    logits: np.ndarray,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    config: PairwiseStackConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the fixed L2 logistic model on mirrored standardized differences."""

    difference = np.asarray(
        logits[positive_rows] - logits[negative_rows], dtype=np.float32
    )
    scale = np.sqrt(
        np.mean(difference * difference, axis=0, dtype=np.float64)
    ).astype(np.float32)
    np.maximum(scale, config.minimum_pair_scale, out=scale)
    difference /= scale
    mirrored = np.concatenate((difference, -difference), axis=0)
    labels = np.concatenate(
        (
            np.ones(difference.shape[0], dtype=np.uint8),
            np.zeros(difference.shape[0], dtype=np.uint8),
        )
    )
    model = LogisticRegression(
        C=config.regularization_c,
        penalty="l2",
        fit_intercept=False,
        solver=config.solver,
        max_iter=config.maximum_iterations,
        tol=config.tolerance,
        n_jobs=1,
    )
    started = time.perf_counter()
    model.fit(mirrored, labels)
    seconds = time.perf_counter() - started
    standardized_coefficient = np.asarray(model.coef_[0], dtype=np.float64)
    raw_coefficient = standardized_coefficient / scale.astype(np.float64)
    if not (
        np.all(np.isfinite(scale))
        and np.all(np.isfinite(standardized_coefficient))
        and np.all(np.isfinite(raw_coefficient))
    ):
        raise ValueError("pairwise model emitted non-finite parameters")
    detail = {
        "fit_seconds": float(seconds),
        "solver_iterations": int(np.asarray(model.n_iter_).max()),
        "converged_before_iteration_limit": bool(
            int(np.asarray(model.n_iter_).max()) < config.maximum_iterations
        ),
        "negative_raw_coefficients": int(np.sum(raw_coefficient < 0.0)),
        "positive_raw_coefficients": int(np.sum(raw_coefficient > 0.0)),
        "zero_raw_coefficients": int(np.sum(raw_coefficient == 0.0)),
    }
    return scale, standardized_coefficient, detail


def predict_margin(
    logits: np.ndarray,
    positions: np.ndarray,
    scale: np.ndarray,
    standardized_coefficient: np.ndarray,
    chunk_rows: int,
) -> np.ndarray:
    """Predict a ranking margin without materializing an entire held fold."""

    rows = np.asarray(positions, dtype=np.int64)
    output = np.empty(rows.size, dtype=np.float32)
    coefficient = np.asarray(standardized_coefficient, dtype=np.float64)
    denominator = np.asarray(scale, dtype=np.float64)
    for left in range(0, rows.size, chunk_rows):
        right = min(left + chunk_rows, rows.size)
        block = np.asarray(logits[rows[left:right]], dtype=np.float64)
        output[left:right] = ((block / denominator) @ coefficient).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("pairwise stack prediction contains NaN or infinity")
    return output


def _score_summary(scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("score summary requires finite values")
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


def _fit_contract(
    config: PairwiseStackConfig,
    data: StackData,
    *,
    role: str,
    held_fold: int | None,
    pair_seed: int,
) -> dict[str, Any]:
    sources = [
        {
            "name": name,
            "path": path.relative_to(config.project_root).as_posix(),
            "sha256": digest,
        }
        for name, path, digest in zip(
            data.feature_names, data.source_paths, data.source_hashes
        )
    ]
    return {
        "experiment_version": PAIRWISE_STACK_VERSION,
        "role": role,
        "held_fold": held_fold,
        "pair_seed": pair_seed,
        "pairs": config.pairs_per_fit,
        "regularization_c": config.regularization_c,
        "solver": config.solver,
        "maximum_iterations": config.maximum_iterations,
        "tolerance": config.tolerance,
        "logit_epsilon": config.logit_epsilon,
        "minimum_pair_scale": config.minimum_pair_scale,
        "feature_names": data.feature_names,
        "sources": sources,
        "model5_selection_hash": data.model5_selection_hash,
        "model6_protocol_hash": data.model6_protocol_hash,
        "outer_holdout_read": False,
    }


def _load_fold_checkpoint(
    directory: Path,
    held_fold: int,
    contract_hash: str,
    expected_rows: int,
    expected_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    report_path = directory / f"fold_{held_fold}.json"
    prediction_path = directory / f"fold_{held_fold}.margin.f32.npy"
    coefficient_path = directory / f"fold_{held_fold}.coefficients.npz"
    if not any(path.exists() for path in (report_path, prediction_path, coefficient_path)):
        return None
    if not all(path.exists() for path in (report_path, prediction_path, coefficient_path)):
        raise ValueError(f"incomplete pairwise checkpoint for fold {held_fold}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("contract_hash") != contract_hash:
        raise ValueError(f"stale pairwise checkpoint: {report_path}")
    prediction = np.load(prediction_path, allow_pickle=False)
    with np.load(coefficient_path, allow_pickle=False) as saved:
        scale = saved["scale"].astype(np.float64)
        coefficient = saved["standardized_coefficient"].astype(np.float64)
    if prediction.shape != (expected_rows,):
        raise ValueError(f"wrong checkpoint prediction shape: {prediction_path}")
    if scale.shape != (expected_features,) or coefficient.shape != (expected_features,):
        raise ValueError(f"wrong checkpoint coefficient shape: {coefficient_path}")
    if not all(np.all(np.isfinite(value)) for value in (prediction, scale, coefficient)):
        raise ValueError(f"non-finite pairwise checkpoint: fold {held_fold}")
    return prediction.astype(np.float32), scale, coefficient, report


def _fit_fold(
    config: PairwiseStackConfig,
    data: StackData,
    held_fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    score_positions = np.flatnonzero(data.row_fold == held_fold)
    training_mask = data.row_fold != held_fold
    pair_seed = config.fold_pair_seed + 100 * held_fold
    contract = _fit_contract(
        config,
        data,
        role="cross_fitted_score_fold",
        held_fold=held_fold,
        pair_seed=pair_seed,
    )
    contract_hash = _hash_json(contract)
    directory = config.artifact_path / "pairwise_cv"
    checkpoint = _load_fold_checkpoint(
        directory,
        held_fold,
        contract_hash,
        score_positions.size,
        len(data.feature_names),
    )
    if checkpoint is not None:
        print(
            f"pairwise stack fold {held_fold}: checkpoint "
            f"TSAUC={checkpoint[3]['tsroc']:.6f}",
            flush=True,
        )
        return checkpoint

    positive, negative, sampling = sample_time_matched_pairs(
        data.target,
        data.time_online,
        training_mask,
        config.pairs_per_fit,
        pair_seed,
    )
    if np.any(data.row_fold[positive] == held_fold) or np.any(
        data.row_fold[negative] == held_fold
    ):
        raise AssertionError("held-fold rows entered pairwise fitting")
    scale, coefficient, fit_detail = fit_pairwise_model(
        data.logits, positive, negative, config
    )
    prediction = predict_margin(
        data.logits,
        score_positions,
        scale,
        coefficient,
        config.prediction_chunk_rows,
    )
    score = float(
        tsauc(
            data.target[score_positions],
            prediction,
            data.time_online[score_positions],
        )
    )
    report = {
        **contract,
        "contract_hash": contract_hash,
        "training_folds": sorted(
            set(data.row_fold[training_mask].astype(int).tolist())
        ),
        "score_rows": int(score_positions.size),
        "score_series": int(
            np.unique(data.series_position[score_positions]).size
        ),
        "sampling": sampling,
        "fit": fit_detail,
        "tsroc": score,
    }
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_npy(directory / f"fold_{held_fold}.margin.f32.npy", prediction)
    _atomic_npz(
        directory / f"fold_{held_fold}.coefficients.npz",
        feature_names=np.asarray(data.feature_names, dtype=np.str_),
        scale=scale.astype(np.float64),
        standardized_coefficient=coefficient.astype(np.float64),
        raw_logit_coefficient=(coefficient / scale).astype(np.float64),
    )
    _atomic_json(directory / f"fold_{held_fold}.json", report)
    print(
        f"pairwise stack fold {held_fold}: TSAUC={score:.6f}", flush=True
    )
    return prediction, scale, coefficient, report


def _fit_deployment(
    config: PairwiseStackConfig, data: StackData
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    directory = config.artifact_path / "deployment"
    report_path = directory / "manifest.json"
    coefficient_path = directory / "coefficients.npz"
    contract = _fit_contract(
        config,
        data,
        role="all_development_deployment_fit",
        held_fold=None,
        pair_seed=config.deployment_pair_seed,
    )
    contract_hash = _hash_json(contract)
    if report_path.exists() or coefficient_path.exists():
        if not report_path.exists() or not coefficient_path.exists():
            raise ValueError("incomplete pairwise deployment checkpoint")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise ValueError("stale pairwise deployment checkpoint")
        with np.load(coefficient_path, allow_pickle=False) as saved:
            scale = saved["scale"].astype(np.float64)
            coefficient = saved["standardized_coefficient"].astype(np.float64)
        return scale, coefficient, report

    eligible = np.ones(data.target.size, dtype=bool)
    positive, negative, sampling = sample_time_matched_pairs(
        data.target,
        data.time_online,
        eligible,
        config.pairs_per_fit,
        config.deployment_pair_seed,
    )
    scale, coefficient, fit_detail = fit_pairwise_model(
        data.logits, positive, negative, config
    )
    raw = coefficient / scale
    report = {
        **contract,
        "contract_hash": contract_hash,
        "development_rows": int(data.target.size),
        "development_series": data.development_series,
        "sampling": sampling,
        "fit": fit_detail,
        "inference": (
            "clip each primitive probability to [1e-6,1-1e-6], take logits, "
            "then dot with raw_logit_coefficient; no intercept"
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        coefficient_path,
        feature_names=np.asarray(data.feature_names, dtype=np.str_),
        scale=scale.astype(np.float64),
        standardized_coefficient=coefficient.astype(np.float64),
        raw_logit_coefficient=raw.astype(np.float64),
    )
    _atomic_json(report_path, report)
    return scale, coefficient, report


def run(config: PairwiseStackConfig) -> dict[str, Any]:
    """Run/resume cross-fitting, fit deployment coefficients, and report deltas."""

    data = load_stack_data(config)
    source_records = [
        {
            "name": name,
            "path": path.relative_to(config.project_root).as_posix(),
            "sha256": digest,
        }
        for name, path, digest in zip(
            data.feature_names, data.source_paths, data.source_hashes
        )
    ]
    manifest = {
        "experiment_version": PAIRWISE_STACK_VERSION,
        "protocol": config.protocol(),
        "protocol_hash": _hash_json(config.protocol()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "development_rows": int(data.target.size),
        "development_series": data.development_series,
        "fold_rows": {
            str(fold): int(np.sum(data.row_fold == fold))
            for fold in range(config.folds)
        },
        "primitive_sources": source_records,
        "base_model_selection_was_not_renested": True,
        "level_two_oof_is_exploratory": True,
        "nesting_caveat": (
            "For a meta-held fold h, a primitive OOF input on training fold f "
            "was produced by a base voter excluding f but potentially trained "
            "on h. A confirmatory estimate requires refitting every base voter "
            "inside each meta-held split."
        ),
        "outer_holdout_read": False,
    }
    _write_or_validate(config.artifact_path / "experiment_manifest.json", manifest)

    oof = np.empty(data.target.size, dtype=np.float32)
    fold_scales = []
    fold_coefficients = []
    fold_reports = []
    pairwise_scores = []
    for held_fold in range(config.folds):
        score_positions = np.flatnonzero(data.row_fold == held_fold)
        prediction, scale, coefficient, report = _fit_fold(
            config, data, held_fold
        )
        oof[score_positions] = prediction
        fold_scales.append(scale)
        fold_coefficients.append(coefficient / scale)
        fold_reports.append(report)
        pairwise_scores.append(float(report["tsroc"]))
    if not np.all(np.isfinite(oof)):
        raise AssertionError("cross-fitted pairwise OOF is incomplete")
    _atomic_npy(config.artifact_path / "development_oof.margin.f32.npy", oof)

    _, deployment_coefficient, deployment_report = _fit_deployment(config, data)
    with np.load(
        config.artifact_path / "deployment/coefficients.npz", allow_pickle=False
    ) as saved:
        deployment_raw = saved["raw_logit_coefficient"].astype(np.float64)
    raw_matrix = np.vstack(fold_coefficients)
    coefficient_report = []
    for index, name in enumerate(data.feature_names):
        values = raw_matrix[:, index]
        coefficient_report.append(
            {
                "feature": name,
                "deployment_raw_logit_coefficient": float(deployment_raw[index]),
                "fold_mean_raw_logit_coefficient": float(values.mean()),
                "fold_standard_deviation": float(values.std(ddof=1)),
                "fold_minimum": float(values.min()),
                "fold_maximum": float(values.max()),
                "same_sign_in_all_folds": bool(
                    np.all(values > 0.0) or np.all(values < 0.0)
                ),
            }
        )
    coefficient_report.sort(
        key=lambda row: -abs(row["deployment_raw_logit_coefficient"])
    )

    pairwise_summary = {
        **_score_summary(pairwise_scores),
        "pooled_development_oof": float(
            tsauc(data.target, oof, data.time_online)
        ),
    }
    # Read Model 6's already-frozen development score contract for a paired
    # fold diagnostic. This JSON contains no holdout predictions or labels and
    # is never an input to the fitted stack.
    frozen_path = config.model6_path / "frozen_contract.json"
    _development_only_path(frozen_path, config.project_root)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    baseline = frozen["development_nested_tsroc"]
    baseline_scores = np.asarray(baseline["fold_scores"], dtype=np.float64)
    differences = np.asarray(pairwise_scores, dtype=np.float64) - baseline_scores
    paired_se = float(fold_standard_error(differences))
    comparison = {
        "baseline_name": "model6_frozen_global",
        "baseline_fold_scores": baseline_scores.tolist(),
        "baseline_fold_mean": float(baseline_scores.mean()),
        "baseline_fold_standard_error": float(fold_standard_error(baseline_scores)),
        "paired_fold_differences": differences.tolist(),
        "mean_gain": float(differences.mean()),
        "paired_fold_gain_standard_error": paired_se,
        "robust_gain_mean_minus_half_se": float(
            differences.mean() - config.uncertainty_penalty * paired_se
        ),
        "fold_wins": int(np.sum(differences > 0.0)),
        "note": (
            "Paired five-fold diagnostic only; a complete-series paired "
            "bootstrap is still required before replacing the deployed model."
        ),
    }
    summary = {
        "experiment_version": PAIRWISE_STACK_VERSION,
        "stage": "fixed_cross_fitted_pairwise_primitive_stack",
        "protocol_hash": manifest["protocol_hash"],
        "pairwise_stack": pairwise_summary,
        "comparison_to_model6": comparison,
        "fold_reports": fold_reports,
        "coefficient_stability": coefficient_report,
        "deployment": deployment_report,
        "oof_prediction_path": (
            config.artifact_path / "development_oof.margin.f32.npy"
        ).relative_to(config.project_root).as_posix(),
        "deployment_coefficient_path": (
            config.artifact_path / "deployment/coefficients.npz"
        ).relative_to(config.project_root).as_posix(),
        "level_two_oof_is_exploratory": True,
        "upstream_model5_bank_selection_reused_development_folds": True,
        "not_fully_nested_base_refit": True,
        "interpretation": (
            "The meta-held rows do not enter pair construction, scaling, or "
            "logistic fitting. However, primitive OOF values on the other folds "
            "can come from base voters trained on the meta-held fold. Use this "
            "for architecture screening only; a confirmatory score requires "
            "base-model refits nested inside every meta-held fold."
        ),
        "outer_holdout_read": False,
        "paired_complete_series_bootstrap_pending": True,
    }
    _atomic_json(config.artifact_path / "summary.json", summary)
    return summary


def status(config: PairwiseStackConfig) -> dict[str, Any]:
    """Report checkpoint presence without loading labels or any holdout artifact."""

    fold_directory = config.artifact_path / "pairwise_cv"
    return {
        "run": config.artifact_directory,
        "manifest": (config.artifact_path / "experiment_manifest.json").exists(),
        "completed_folds": [
            fold
            for fold in range(config.folds)
            if (fold_directory / f"fold_{fold}.json").exists()
            and (fold_directory / f"fold_{fold}.margin.f32.npy").exists()
            and (fold_directory / f"fold_{fold}.coefficients.npz").exists()
        ],
        "deployment": (config.artifact_path / "deployment/manifest.json").exists(),
        "summary": (config.artifact_path / "summary.json").exists(),
        "outer_holdout_read": False,
    }


def _project_root() -> Path:
    return PROJECT_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"), nargs="?", default="run")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--model6-run", default=DEFAULT_MODEL6_RUN)
    arguments = parser.parse_args(argv)
    config = make_config(arguments.project_root, arguments.model6_run)
    result = run(config) if arguments.command == "run" else status(config)
    print(json.dumps(_json_ready(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
