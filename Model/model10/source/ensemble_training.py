"""Self-contained training for the selected core/candidate submission.

This module is used by the no-pretrained-model bundle.  It consumes only the
labeled training tuples supplied to Crunch's ``train`` function, constructs
causal features from each series' clean history and online prefix, fits the
two independently selected LightGBM families, and writes the relative-path
ensemble expected by ``solution.infer``.

No test file, protected archive, precomputed feature cache, fold prediction,
or pretrained model is read. All temporary and final artifacts are written
strictly inside the model directory because Crunch mounts the code directory
read-only while granting write access to ``resources/``.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import lightgbm as lgb
import numpy as np

from feature_pipeline import (
    CANDIDATE_BANKS,
    CORE_BANKS,
    compute_features_offline,
    feature_names,
)
from train_cv import (
    DEFAULT_MODEL_PARAMETERS,
    deterministic_sample_mask,
    pair_weights_from_reference,
)


TRAIN_SAMPLE_FRACTION = 0.20
TRAIN_SAMPLE_SEED = 99173
MODEL_SEED = 51413
PROGRESS_EVERY = 25


@dataclass(frozen=True)
class ComponentSpec:
    """One independently fitted family in the final logit ensemble."""

    name: str
    banks: tuple[str, ...]
    boosting_rounds: int
    weight: float = 0.5
    selection_file: str | None = None
    selection_method: str = "all"


SELECTION_FILES = {
    "core": "core_feature_selection.npz",
    "candidate": "candidate_feature_selection.npz",
}
SELECTION_METHODS = {
    "core": "consensus_top_768",
    "candidate": "gain_corr_0p98",
}
SELECTION_CONTRACT_HASH = (
    "e7154880585cb472bbed94079048312641d5b4f03d6dfc912ac4a8e908861a8f"
)


def _has_selected_feature_contract() -> bool:
    """Return whether this code bundle contains the frozen selection contract."""

    code_directory = Path(__file__).resolve().parent
    present = {
        name: (code_directory / filename).is_file()
        for name, filename in SELECTION_FILES.items()
    }
    if any(present.values()) and not all(present.values()):
        missing = [name for name, exists in present.items() if not exists]
        raise RuntimeError(
            "incomplete selected-feature bundle; missing contract for "
            + ", ".join(missing)
        )
    return all(present.values())


# The bundle builder includes both frozen NPZ contracts only for submission 3.
# The unbundled source and submission 2 retain their all-feature specifications.
if _has_selected_feature_contract():
    COMPONENTS = (
        ComponentSpec(
            "core",
            tuple(CORE_BANKS),
            boosting_rounds=101,
            weight=0.55,
            selection_file=SELECTION_FILES["core"],
            selection_method=SELECTION_METHODS["core"],
        ),
        ComponentSpec(
            "candidate",
            tuple(CANDIDATE_BANKS),
            boosting_rounds=217,
            weight=0.45,
            selection_file=SELECTION_FILES["candidate"],
            selection_method=SELECTION_METHODS["candidate"],
        ),
    )
else:
    # Median iterations selected for the original all-feature submission 2.
    COMPONENTS = (
        ComponentSpec("core", tuple(CORE_BANKS), boosting_rounds=183),
        ComponentSpec("candidate", tuple(CANDIDATE_BANKS), boosting_rounds=103),
    )


@dataclass
class PreparedTrainingData:
    """Shared row selection and labels reused by both feature families."""

    records: list[tuple[int, np.ndarray, np.ndarray, int | None]]
    masks: list[np.ndarray]
    labels: np.ndarray
    online_time: np.ndarray
    weights: np.ndarray
    sample_rows: int


def _target(length: int, tau_index: int | None) -> np.ndarray:
    if tau_index is None or int(tau_index) < 0:
        return np.zeros(length, dtype=np.uint8)
    return (np.arange(length) >= int(tau_index)).astype(np.uint8)


def _normalize_records(
    datasets: Iterable[tuple],
) -> list[tuple[int, np.ndarray, np.ndarray, int | None]]:
    """Consume Crunch's one-pass train iterator and validate its shape."""

    records: list[tuple[int, np.ndarray, np.ndarray, int | None]] = []
    seen_ids: set[int] = set()
    for position, item in enumerate(datasets):
        if len(item) != 4:
            raise ValueError(
                "training records must be (id, history, online, tau_index)"
            )
        dataset_id, historical, online, tau_index = item
        dataset_id = int(dataset_id)
        if dataset_id in seen_ids:
            raise ValueError(f"duplicate training series id: {dataset_id}")
        seen_ids.add(dataset_id)

        historical_array = np.asarray(historical, dtype=np.float64)
        online_array = np.asarray(online, dtype=np.float64)
        if historical_array.ndim != 1 or online_array.ndim != 1:
            raise ValueError(f"series {dataset_id} is not one-dimensional")
        if historical_array.size == 0 or online_array.size == 0:
            raise ValueError(f"series {dataset_id} contains an empty period")

        normalized_tau = None if tau_index is None else int(tau_index)
        if normalized_tau is not None and normalized_tau >= online_array.size:
            raise ValueError(
                f"series {dataset_id} has tau_index outside its online period"
            )
        records.append(
            (
                dataset_id,
                historical_array,
                online_array,
                normalized_tau,
            )
        )

    if not records:
        raise ValueError("training dataset is empty")
    return records


def _prepare_training_data(datasets: Iterable[tuple]) -> PreparedTrainingData:
    """Select deterministic training rows using only labels from train data."""

    records = _normalize_records(datasets)
    masks: list[np.ndarray] = []
    labels_by_series: list[np.ndarray] = []
    time_by_series: list[np.ndarray] = []

    for dataset_id, _, online, tau_index in records:
        online_time = np.arange(online.size, dtype=np.int16)
        labels = _target(online.size, tau_index)
        mask = deterministic_sample_mask(
            np.full(online.size, dataset_id, dtype=np.int64),
            online_time,
            TRAIN_SAMPLE_FRACTION,
            TRAIN_SAMPLE_SEED,
        )
        masks.append(mask)
        labels_by_series.append(labels)
        time_by_series.append(online_time)

    complete_labels = np.concatenate(labels_by_series)
    complete_time = np.concatenate(time_by_series)
    sampled_labels = np.concatenate(
        [labels[mask] for labels, mask in zip(labels_by_series, masks)]
    )
    sampled_time = np.concatenate(
        [time[mask] for time, mask in zip(time_by_series, masks)]
    )
    if sampled_labels.size == 0 or np.unique(sampled_labels).size != 2:
        raise ValueError("sampled training rows must contain both target classes")

    weights = pair_weights_from_reference(
        sampled_labels,
        sampled_time,
        complete_labels,
        complete_time,
    )
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        raise ValueError("training row weights are invalid")

    return PreparedTrainingData(
        records=records,
        masks=masks,
        labels=sampled_labels,
        online_time=sampled_time,
        weights=weights,
        sample_rows=int(sampled_labels.size),
    )


def _model_parameters() -> dict[str, object]:
    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    parameters.update(
        {
            "seed": MODEL_SEED,
            "feature_fraction_seed": MODEL_SEED + 100,
            "bagging_seed": MODEL_SEED + 200,
            "data_random_seed": MODEL_SEED + 300,
        }
    )
    return parameters


def _feature_contract(
    specification: ComponentSpec,
) -> tuple[list[str], np.ndarray]:
    """Load and strictly validate one all-feature or selected-feature contract.

    The NPZ contains no model weights or observations.  It freezes only the
    train-derived source column numbers and matching feature names selected
    before the locked test was opened.
    """

    available_names = feature_names(specification.banks)
    if specification.selection_file is None:
        return available_names, np.arange(len(available_names), dtype=np.int32)

    contract_path = Path(__file__).resolve().with_name(
        specification.selection_file
    )
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing feature-selection contract: {contract_path}")
    with np.load(contract_path, allow_pickle=False) as contract:
        if set(contract.files) != {"columns", "names"}:
            raise ValueError(
                f"unexpected arrays in feature-selection contract: {contract_path}"
            )
        columns = np.asarray(contract["columns"], dtype=np.int32)
        selected_names = [str(name) for name in contract["names"]]

    if columns.ndim != 1 or columns.size == 0:
        raise ValueError("selected feature columns must be a nonempty vector")
    if len(selected_names) != columns.size:
        raise ValueError("selected feature names and columns differ in length")
    if np.any(columns < 0) or np.any(columns >= len(available_names)):
        raise ValueError("selected feature contract has an out-of-range column")
    if len(set(columns.tolist())) != columns.size:
        raise ValueError("selected feature contract has duplicate columns")
    expected_names = [available_names[index] for index in columns]
    if selected_names != expected_names:
        raise ValueError(
            "selected feature names do not match the submitted detector code"
        )
    return selected_names, columns


def _fit_component(
    prepared: PreparedTrainingData,
    specification: ComponentSpec,
    output_directory: Path,
) -> None:
    """Materialize one sampled family, fit it, and save its full contract."""

    names, selected_columns = _feature_contract(specification)
    matrix = np.empty(
        (prepared.sample_rows, len(names)),
        dtype=np.float32,
    )
    cursor = 0
    total_series = len(prepared.records)
    for position, (record, mask) in enumerate(
        zip(prepared.records, prepared.masks), start=1
    ):
        _, historical, online, _ = record
        features = compute_features_offline(
            historical,
            online,
            specification.banks,
        )
        count = int(mask.sum())
        matrix[cursor : cursor + count] = features[mask][:, selected_columns]
        cursor += count
        if position % PROGRESS_EVERY == 0 or position == total_series:
            print(
                f"{specification.name}: features {position:,}/{total_series:,} "
                f"series ({cursor:,}/{prepared.sample_rows:,} sampled rows)",
                flush=True,
            )
    if cursor != prepared.sample_rows:
        raise AssertionError("sample row accounting differs between families")

    parameters = _model_parameters()
    dataset = lgb.Dataset(
        matrix,
        label=prepared.labels,
        weight=prepared.weights,
        feature_name=names,
        params=parameters,
        free_raw_data=True,
    )
    dataset.construct()
    del matrix
    gc.collect()

    print(
        f"{specification.name}: fitting {specification.boosting_rounds} rounds",
        flush=True,
    )
    booster = lgb.train(
        parameters,
        dataset,
        num_boost_round=specification.boosting_rounds,
    )
    if booster.num_feature() != len(names):
        raise AssertionError("trained model feature count is inconsistent")

    component_directory = output_directory / specification.name
    component_directory.mkdir(parents=True)
    model_path = component_directory / "final.txt"
    booster.save_model(model_path)
    manifest = {
        "banks": list(specification.banks),
        "feature_names": names,
        "feature_columns": selected_columns.tolist(),
        "source_feature_count": len(feature_names(specification.banks)),
        "feature_selection": {
            "method": specification.selection_method,
            "contract_hash": (
                SELECTION_CONTRACT_HASH
                if specification.selection_file is not None
                else None
            ),
            "selection_source": (
                "train-only ranking; validation-only choice; locked test excluded"
                if specification.selection_file is not None
                else "all source-family columns"
            ),
        },
        "sample_fraction": TRAIN_SAMPLE_FRACTION,
        "sample_seed": TRAIN_SAMPLE_SEED,
        "parameters": parameters,
        "training_source": "Crunch train() labeled datasets only",
        "final_model": {
            "path": model_path.name,
            "iterations": specification.boosting_rounds,
            "training_rows": prepared.sample_rows,
            "model_seed": MODEL_SEED,
        },
    }
    (component_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    del booster, dataset
    gc.collect()


def _write_ensemble_manifest(output_directory: Path) -> None:
    manifest = {
        "ensemble_method": "logit",
        "components": [
            {
                "directory": specification.name,
                "weight": specification.weight,
            }
            for specification in COMPONENTS
        ],
        "postprocess_penalty": None,
        "training_source": "Crunch train() labeled datasets only",
        "feature_selection_contract_hash": (
            SELECTION_CONTRACT_HASH
            if all(item.selection_file is not None for item in COMPONENTS)
            else None
        ),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _promote_staged_model(staging: Path, destination: Path) -> None:
    """Publish staged files without writing to the read-only code directory.

    Component weights and contracts are replaced first. The top-level
    ensemble manifest is replaced last, so a successful ``train`` return
    always leaves a complete model contract at the exact resources path.
    """

    if destination.is_symlink():
        raise ValueError("refusing to write through a symlinked model directory")
    destination.mkdir(parents=True, exist_ok=True)
    for specification in COMPONENTS:
        source_component = staging / specification.name
        target_component = destination / specification.name
        target_component.mkdir(exist_ok=True)
        os.replace(
            source_component / "final.txt",
            target_component / "final.txt",
        )
        os.replace(
            source_component / "manifest.json",
            target_component / "manifest.json",
        )
    os.replace(staging / "manifest.json", destination / "manifest.json")


def train_ensemble(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Fit both selected families from scratch and atomically publish them."""

    destination = Path(model_directory_path)
    destination.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_training_data(datasets)
    print(
        f"training two-family ensemble from {len(prepared.records):,} series "
        f"and {prepared.sample_rows:,} sampled online rows",
        flush=True,
    )

    with TemporaryDirectory(
        prefix=".training_",
        dir=destination,
    ) as temporary_directory:
        staging = Path(temporary_directory)
        for specification in COMPONENTS:
            _fit_component(prepared, specification, staging)
        _write_ensemble_manifest(staging)

        expected = {
            staging / "manifest.json",
            *(
                staging / specification.name / "manifest.json"
                for specification in COMPONENTS
            ),
            *(
                staging / specification.name / "final.txt"
                for specification in COMPONENTS
            ),
        }
        if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
            raise RuntimeError("staged ensemble is incomplete")
        _promote_staged_model(staging, destination)

    print(f"saved trained core+candidate ensemble to {destination}", flush=True)


__all__ = ["COMPONENTS", "train_ensemble"]
