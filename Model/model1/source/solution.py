"""Crunch-compatible training and strictly streaming inference entry points.

The selected deployment is a relative-path bundle containing independently
trained core and candidate boosters and a 50/50 logit ensemble manifest. Use
``build_submission_bundle.py`` to copy only production files and final models;
the multi-gigabyte training caches are never needed by inference. Alternatively
Crunch may call :func:`train`, which reproduces the lower-cost single-family
fallback from its training iterator.

The inference function obeys the corrected real-time protocol:

* the historical segment is scanned once when a series begins;
* exactly one score is yielded after each newly released online value;
* no future online value, final online length, test label, filesystem state,
  or state from another series is used;
* output is deterministic to substantially tighter than the platform's 1e-8
  repeatability tolerance.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import lightgbm as lgb
import numpy as np

from feature_pipeline import (
    CORE_BANKS,
    CoreStreamDetector,
    bank_slices,
    compute_features_offline,
    feature_names,
)
from train_cv import (
    DEFAULT_MODEL_PARAMETERS,
    deterministic_sample_mask,
    pair_weights_from_reference,
)
from postprocess import StreamingDecayedMax


TRAIN_SAMPLE_FRACTION = 0.20
TRAIN_SAMPLE_SEED = 99173
FINAL_MODEL_SEED = 51413
# Updated automatically in the checked-in artifact after CV; this conservative
# default is used only when Crunch is asked to train directly from this file.
FINAL_BOOSTING_ROUNDS = 183


def _target(length: int, tau_index: Optional[int]) -> np.ndarray:
    if tau_index is None or int(tau_index) < 0:
        return np.zeros(length, dtype=np.uint8)
    return (np.arange(length) >= int(tau_index)).astype(np.uint8)


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Fit the deterministic deployment model on labeled training series.

    Crunch uploads the submitted model directory before invoking this function.
    If that directory contains the selected pretrained ensemble, validate it
    and leave it byte-for-byte unchanged.  This is important because replacing
    the ensemble manifest with the lower-cost fallback model would silently
    deploy a weaker model than the one selected by cross-validation.

    ``datasets`` elements must be ``(id, history, online, tau_index)``.  Only a
    deterministic 20% of decision rows is retained for supervised fitting,
    but features for a retained row are still computed from its complete
    causal prefix.  This reduces peak model-training memory without truncating
    histories or leaking the break label into feature construction.
    """

    directory = Path(model_directory_path)
    uploaded_manifest_path = directory / "manifest.json"
    if uploaded_manifest_path.exists():
        try:
            uploaded_manifest = json.loads(uploaded_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid uploaded model manifest: {uploaded_manifest_path}"
            ) from error
        if "components" in uploaded_manifest:
            # Load every component and verify its feature contract/model shape.
            # Do this before touching ``datasets`` so a valid submitted
            # ensemble makes forced first-train a cheap, deterministic no-op.
            _load_ensemble(str(directory))
            return

    records = list(datasets)
    if not records:
        raise ValueError("training dataset is empty")

    ids = np.asarray([int(record[0]) for record in records], dtype=np.int64)
    online_values = [np.asarray(record[2], dtype=np.float64) for record in records]
    tau = [None if record[3] is None else int(record[3]) for record in records]
    online_lengths = np.asarray([values.size for values in online_values])

    masks: list[np.ndarray] = []
    full_time: list[np.ndarray] = []
    full_target: list[np.ndarray] = []
    sample_rows = 0
    for dataset_id, length, break_index in zip(ids, online_lengths, tau):
        step = np.arange(length, dtype=np.int16)
        labels = _target(int(length), break_index)
        mask = deterministic_sample_mask(
            np.full(length, dataset_id, dtype=np.int64),
            step,
            TRAIN_SAMPLE_FRACTION,
            TRAIN_SAMPLE_SEED,
        )
        masks.append(mask)
        full_time.append(step)
        full_target.append(labels)
        sample_rows += int(mask.sum())

    names = feature_names(CORE_BANKS)
    x_sample = np.empty((sample_rows, len(names)), dtype=np.float32)
    y_sample = np.empty(sample_rows, dtype=np.uint8)
    time_sample = np.empty(sample_rows, dtype=np.int16)
    cursor = 0
    for record, stream, mask, labels, step in zip(
        records, online_values, masks, full_target, full_time
    ):
        history = np.asarray(record[1], dtype=np.float64)
        features = compute_features_offline(history, stream, CORE_BANKS)
        count = int(mask.sum())
        x_sample[cursor : cursor + count] = features[mask]
        y_sample[cursor : cursor + count] = labels[mask]
        time_sample[cursor : cursor + count] = step[mask]
        cursor += count
    if cursor != sample_rows:
        raise AssertionError("sample row accounting error")

    target_all = np.concatenate(full_target)
    time_all = np.concatenate(full_time)
    weights = pair_weights_from_reference(
        y_sample, time_sample, target_all, time_all
    )
    training = lgb.Dataset(
        x_sample,
        label=y_sample,
        weight=weights,
        feature_name=names,
        free_raw_data=True,
    )
    training.construct()
    del x_sample, y_sample, time_sample, weights, target_all, time_all
    gc.collect()

    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    parameters.update(
        {
            "seed": FINAL_MODEL_SEED,
            "feature_fraction_seed": FINAL_MODEL_SEED + 100,
            "bagging_seed": FINAL_MODEL_SEED + 200,
            "data_random_seed": FINAL_MODEL_SEED + 300,
        }
    )
    booster = lgb.train(
        parameters, training, num_boost_round=FINAL_BOOSTING_ROUNDS
    )

    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "final.txt"
    booster.save_model(model_path)
    manifest = {
        "banks": list(CORE_BANKS),
        "feature_names": names,
        "final_model": {
            "path": model_path.name,
            "iterations": FINAL_BOOSTING_ROUNDS,
            "training_rows": sample_rows,
            "model_seed": FINAL_MODEL_SEED,
        },
        "sample_fraction": TRAIN_SAMPLE_FRACTION,
        "sample_seed": TRAIN_SAMPLE_SEED,
        "parameters": parameters,
        "postprocess_penalty": None,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))


@dataclass
class LoadedComponent:
    """One independently trained feature/model family in the ensemble."""

    banks: tuple[str, ...]
    boosters: list[lgb.Booster]
    weight: float
    time_min: int
    time_max: Optional[int]
    columns: np.ndarray | None = None


def _load_single_component(
    directory: Path,
    weight: float = 1.0,
    time_min: int = 0,
    time_max: Optional[int] = None,
) -> LoadedComponent:
    """Load a conventional CV/final model directory as one component."""

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing trained-model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    banks = tuple(manifest["banks"])
    expected_names = feature_names(banks)
    if manifest["feature_names"] != expected_names:
        raise ValueError("model feature contract differs from the submitted code")

    final = manifest.get("final_model")
    if final is not None and (directory / final["path"]).exists():
        paths = [directory / final["path"]]
    else:
        # A completed CV directory is itself deployable.  Averaging fold
        # probabilities is slower but provides a safe fallback before fit-final.
        paths = sorted(directory.glob("fold_*.txt"))
    if not paths:
        raise FileNotFoundError(f"no LightGBM model files found in {directory}")
    boosters = [lgb.Booster(model_file=str(path)) for path in paths]
    expected_columns = len(expected_names)
    if any(model.num_feature() != expected_columns for model in boosters):
        raise ValueError("model feature count differs from the detector output")
    return LoadedComponent(
        banks=banks,
        boosters=boosters,
        weight=float(weight),
        time_min=int(time_min),
        time_max=None if time_max is None else int(time_max),
    )


def _load_ensemble(
    model_directory_path: str,
) -> tuple[tuple[str, ...], list[LoadedComponent], str, Optional[float]]:
    """Load either a legacy single model or a relative-path ensemble bundle."""

    directory = Path(model_directory_path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing trained-model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    specifications = manifest.get("components")
    if specifications is None:
        components = [_load_single_component(directory)]
        method = "probability"
        penalty = manifest.get("postprocess_penalty")
    else:
        if not specifications:
            raise ValueError("ensemble manifest has no components")
        components = []
        for specification in specifications:
            component_directory = directory / specification["directory"]
            components.append(
                _load_single_component(
                    component_directory,
                    weight=float(specification["weight"]),
                    time_min=int(specification.get("time_min", 0)),
                    time_max=specification.get("time_max"),
                )
            )
        method = manifest.get("ensemble_method", "logit")
        if method not in ("logit", "probability"):
            raise ValueError("ensemble_method must be logit or probability")
        penalty = manifest.get("postprocess_penalty")

    if penalty is not None:
        penalty = float(penalty)
    union_banks: list[str] = []
    for component in components:
        for bank in component.banks:
            if bank not in union_banks:
                union_banks.append(bank)
    slices = bank_slices(union_banks)
    for component in components:
        component.columns = np.concatenate(
            [
                np.arange(slices[bank].start, slices[bank].stop, dtype=np.int32)
                for bank in component.banks
            ]
        )
    return tuple(union_banks), components, method, penalty


def _test_item_parts(item: tuple) -> tuple[Sequence[float], Iterable[float]]:
    """Accept both documented local/cloud tuple layouts."""

    if len(item) == 2:
        historical, online = item
    elif len(item) == 3:
        _, historical, online = item
    else:
        raise ValueError(
            "test dataset elements must be (history, online) or "
            "(id, history, online)"
        )
    return historical, online


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield one break probability for every newly arrived online value."""

    banks, components, method, penalty = _load_ensemble(model_directory_path)
    yield  # Required ready signal; it intentionally yields ``None`` once.

    for item in datasets:
        historical, online = _test_item_parts(item)
        detector = CoreStreamDetector(np.asarray(historical, dtype=np.float64), banks)
        persistence = StreamingDecayedMax(penalty)
        for step, value in enumerate(online):
            row = detector.step(float(value)).reshape(1, -1)
            weighted_score = 0.0
            total_weight = 0.0
            for component in components:
                if step < component.time_min or (
                    component.time_max is not None and step >= component.time_max
                ):
                    continue
                component_row = row[:, component.columns]
                component_probability = 0.0
                for booster in component.boosters:
                    component_probability += float(
                        booster.predict(
                            component_row,
                            num_iteration=booster.current_iteration(),
                            num_threads=1,
                        )[0]
                    )
                component_probability /= len(component.boosters)
                if method == "logit":
                    bounded = float(np.clip(component_probability, 1e-7, 1 - 1e-7))
                    score = math.log(bounded) - math.log1p(-bounded)
                else:
                    score = component_probability
                weighted_score += component.weight * score
                total_weight += component.weight
            if total_weight <= 0:
                raise ValueError(f"no ensemble component is active at online step {step}")
            score = weighted_score / total_weight
            if method == "logit":
                if score >= 0:
                    probability = 1.0 / (1.0 + math.exp(-score))
                else:
                    exponential = math.exp(score)
                    probability = exponential / (1.0 + exponential)
            else:
                probability = score
            if not math.isfinite(probability):
                probability = 0.5
            margin = persistence.step(probability)
            if margin >= 0:
                probability = 1.0 / (1.0 + math.exp(-margin))
            else:
                exponential = math.exp(margin)
                probability = exponential / (1.0 + exponential)
            # Keep a tiny margin from exact endpoints without affecting ranks.
            yield float(np.clip(probability, 1e-7, 1.0 - 1e-7))
