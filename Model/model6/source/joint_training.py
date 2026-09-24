"""Train one selected joint LightGBM entirely inside Crunch resources.

This module is copied into ``submission_5`` only if the sealed experiment says
the joint model beats the same-split incumbent.  It bundles no learned weight.
``train_joint_model`` consumes Crunch's labeled iterator once, creates causal
core+candidate+novel features, projects the frozen integer column contract,
fits one LightGBM, and atomically publishes its manifest/model under the
provided writable resources directory.  Temporary matrices are also created
there because Crunch mounts submission code read-only.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import lightgbm as lgb
import numpy as np

from feature_pipeline import (
    CANDIDATE_BANKS,
    CORE_BANKS,
    NOVEL_BANKS,
    compute_features_offline,
    feature_names,
)
from train_cv import deterministic_sample_mask, pair_weights_from_reference


TRAIN_SAMPLE_FRACTION = 0.20
TRAIN_SAMPLE_SEED = 99_173
PROGRESS_EVERY = 25
CONFIG_FILE = "joint_model_config.json"
SELECTION_FILE = "joint_feature_selection.npz"


def _target(length: int, tau_index: int | None) -> np.ndarray:
    if tau_index is None or int(tau_index) < 0:
        return np.zeros(length, dtype=np.uint8)
    return (np.arange(length) >= int(tau_index)).astype(np.uint8)


def _normalize(datasets: Iterable[tuple]) -> list[tuple]:
    records, seen = [], set()
    for item in datasets:
        if len(item) != 4:
            raise ValueError("train record must be (id, history, online, tau_index)")
        dataset_id, history, online, tau_index = item
        dataset_id = int(dataset_id)
        if dataset_id in seen:
            raise ValueError(f"duplicate training series id {dataset_id}")
        seen.add(dataset_id)
        history = np.asarray(history, dtype=np.float64)
        online = np.asarray(online, dtype=np.float64)
        if history.ndim != 1 or online.ndim != 1 or not history.size or not online.size:
            raise ValueError(f"invalid one-dimensional series {dataset_id}")
        tau_index = None if tau_index is None else int(tau_index)
        if tau_index is not None and tau_index >= online.size:
            raise ValueError(f"tau_index outside online period for {dataset_id}")
        records.append((dataset_id, history, online, tau_index))
    if not records:
        raise ValueError("training dataset is empty")
    return records


def _contract() -> tuple[dict[str, object], tuple[str, ...], np.ndarray, list[str]]:
    directory = Path(__file__).resolve().parent
    config_path = directory / CONFIG_FILE
    selection_path = directory / SELECTION_FILE
    if not config_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("submission is missing its frozen joint contract")
    config = json.loads(config_path.read_text())
    banks = tuple(CORE_BANKS + CANDIDATE_BANKS + NOVEL_BANKS)
    if tuple(config["banks"]) != banks:
        raise ValueError("joint config bank order differs from submitted code")
    available = feature_names(banks)
    with np.load(selection_path, allow_pickle=False) as payload:
        if set(payload.files) != {"columns", "names"}:
            raise ValueError("unexpected arrays in joint feature contract")
        columns = payload["columns"].astype(np.int32)
        names = [str(name) for name in payload["names"]]
    if columns.ndim != 1 or not columns.size or len(set(columns.tolist())) != columns.size:
        raise ValueError("joint feature columns must be unique and nonempty")
    if np.any(columns < 0) or np.any(columns >= len(available)):
        raise ValueError("joint feature contract has an out-of-range column")
    if names != [available[int(column)] for column in columns]:
        raise ValueError("joint feature names do not match their source columns")
    return config, banks, columns, names


def train_joint_model(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Fit from labeled train tuples and write only below model_directory_path."""

    destination = Path(model_directory_path)
    if destination.is_symlink():
        raise ValueError("refusing to write through a symlinked resources path")
    destination.mkdir(parents=True, exist_ok=True)
    config, banks, selected_columns, selected_names = _contract()
    records = _normalize(datasets)

    masks, labels_by_series, times_by_series = [], [], []
    sample_rows = 0
    for dataset_id, _, online, tau_index in records:
        step = np.arange(online.size, dtype=np.int16)
        labels = _target(online.size, tau_index)
        mask = deterministic_sample_mask(
            np.full(online.size, dataset_id, dtype=np.int64),
            step, TRAIN_SAMPLE_FRACTION, TRAIN_SAMPLE_SEED,
        )
        masks.append(mask)
        labels_by_series.append(labels)
        times_by_series.append(step)
        sample_rows += int(mask.sum())
    sampled_target = np.concatenate([
        labels[mask] for labels, mask in zip(labels_by_series, masks)
    ])
    sampled_time = np.concatenate([
        step[mask] for step, mask in zip(times_by_series, masks)
    ])
    complete_target = np.concatenate(labels_by_series)
    complete_time = np.concatenate(times_by_series)
    weights = pair_weights_from_reference(
        sampled_target, sampled_time, complete_target, complete_time
    )

    with TemporaryDirectory(prefix=".joint_training_", dir=destination) as temporary:
        staging = Path(temporary)
        matrix_path = staging / "sampled_features.f32.mmap"
        matrix = np.memmap(
            matrix_path, dtype=np.float32, mode="w+",
            shape=(sample_rows, selected_columns.size),
        )
        cursor = 0
        for position, (record, mask) in enumerate(zip(records, masks), start=1):
            _, history, online, _ = record
            features = compute_features_offline(history, online, banks)
            count = int(mask.sum())
            matrix[cursor:cursor + count] = features[mask][:, selected_columns]
            cursor += count
            if position % PROGRESS_EVERY == 0 or position == len(records):
                matrix.flush()
                print(f"joint features {position:,}/{len(records):,} series "
                      f"({cursor:,}/{sample_rows:,} sampled rows)", flush=True)
        if cursor != sample_rows:
            raise AssertionError("sample row accounting mismatch")

        parameters = dict(config["parameters"])
        parameters["num_threads"] = max(1, min(8, os.cpu_count() or 1))
        training = lgb.Dataset(
            matrix, label=sampled_target, weight=weights,
            feature_name=selected_names, params=parameters, free_raw_data=True,
        )
        training.construct()
        del matrix, sampled_target, sampled_time, complete_target, complete_time, weights
        gc.collect()
        rounds = int(config["boosting_rounds"])
        print(f"fitting one joint LightGBM for {rounds} rounds", flush=True)
        booster = lgb.train(parameters, training, num_boost_round=rounds)
        staged_model = staging / "final.txt"
        booster.save_model(staged_model)
        manifest = {
            "banks": list(banks),
            "feature_names": selected_names,
            "feature_columns": selected_columns.tolist(),
            "source_feature_count": len(feature_names(banks)),
            "feature_selection": {
                "method": config["selection_name"],
                "contract_hash": config["frozen_contract_hash"],
                "selection_source": (
                    "train-only rankings, validation-only choice, sealed test after freeze"
                ),
            },
            "sample_fraction": TRAIN_SAMPLE_FRACTION,
            "sample_seed": TRAIN_SAMPLE_SEED,
            "parameters": parameters,
            "training_source": "Crunch train() labeled datasets only",
            "final_model": {
                "path": "final.txt", "iterations": rounds,
                "training_rows": sample_rows,
                "model_seed": int(parameters["seed"]),
            },
        }
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2))
        if not staged_model.stat().st_size or not staged_manifest.stat().st_size:
            raise RuntimeError("staged joint model is incomplete")
        os.replace(staged_model, destination / "final.txt")
        os.replace(staged_manifest, destination / "manifest.json")
        del booster, training
        gc.collect()
    print(f"saved joint model to {destination}", flush=True)


__all__ = ["train_joint_model"]
