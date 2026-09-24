"""Frozen Model 6 deployment with strictly causal streaming inference.

The six retained components are the five score-fold-excluded voters used for
the reported holdout estimate.  Heavy ML libraries and fitted models are
loaded only inside ``infer``: Crunch calls ``train`` in the parent process and
then forks inference workers, so initializing OpenMP-backed libraries during
``train`` can deadlock those workers.  Each online value is consumed once, all
persistence states reset between series, and no future value is inspected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Package imports for repository tests.
    from .compact_runtime import (
        CompactForestBag,
        CompactLinearLogistic,
        load_compact_forest_bag,
        load_compact_linear,
    )
except ImportError:  # Direct imports in Crunch's /context/code directory.
    from compact_runtime import (
        CompactForestBag,
        CompactLinearLogistic,
        load_compact_forest_bag,
        load_compact_linear,
    )


EPSILON = 1e-7
EXPECTED_DEPLOYMENT_PROFILE = "timeout_safe_v1"


def _clip_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(np.clip(value, EPSILON, 1.0 - EPSILON))


def _safe_logit(value: float) -> float:
    return float(_safe_logit_array(np.asarray([value], dtype=np.float64))[0])


def _safe_logit_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Vectorized equivalent of ``_safe_logit`` for model-voter arrays."""

    bounded = np.asarray(values, dtype=np.float64)
    bounded = np.where(np.isfinite(bounded), bounded, 0.5)
    bounded = np.clip(bounded, EPSILON, 1.0 - EPSILON)
    return np.log(bounded) - np.log1p(-bounded)


def _logistic(margin: float) -> float:
    bounded = float(np.clip(margin, -35.0, 35.0))
    return float(1.0 / (1.0 + np.exp(-bounded)))


def _float32_probability(value: float) -> float:
    return float(np.float32(_clip_probability(value)))


def _bag_probabilities(probabilities: Sequence[float]) -> float:
    """Match Model 6's float32 fold-margin bagging exactly."""

    margins = _safe_logit_array(probabilities).astype(np.float32)
    mean_margin = float(np.mean(margins, dtype=np.float32))
    return _float32_probability(_logistic(mean_margin))


def _bag_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    """Vectorized five-voter logit bag for all detector banks."""

    bounded = np.asarray(probabilities, dtype=np.float64)
    if bounded.ndim != 2 or bounded.shape[1] != 5:
        raise ValueError("detector fold probabilities must have five columns")
    margins = _safe_logit_array(bounded).astype(np.float32)
    mean_margin = np.mean(margins, axis=1, dtype=np.float32)
    output = 1.0 / (
        1.0 + np.exp(-np.clip(mean_margin.astype(np.float64), -35.0, 35.0))
    )
    return np.clip(output, EPSILON, 1.0 - EPSILON).astype(np.float32)


def _bag_margin_rows(margins: np.ndarray) -> np.ndarray:
    """Bag native raw margins exactly like Model 6's holdout path."""

    values = np.asarray(margins, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("detector fold margins must have five columns")
    mean_margin = np.mean(values, axis=1, dtype=np.float32)
    return (
        1.0
        / (1.0 + np.exp(-np.clip(mean_margin.astype(np.float64), -35.0, 35.0)))
    ).astype(np.float32)


def _stack_features(probabilities: Sequence[float]) -> np.ndarray:
    """Detector logits followed by Model 6's six agreement features."""

    margins = _safe_logit_array(probabilities).astype(np.float32)
    ordered = np.sort(margins)
    gap = ordered[-1] - ordered[-2] if margins.size > 1 else margins[0]
    extras = np.asarray(
        [
            margins.mean(),
            margins.std(),
            margins.min(),
            margins.max(),
            np.median(margins),
            gap,
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(np.concatenate((margins, extras)), dtype=np.float32)


def _resource_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("resource paths must be relative")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("resource path escapes the submitted model directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing submitted resource: {resolved}")
    return resolved


def _verify_hashes(root: Path, manifest: dict[str, Any]) -> None:
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("submission resource manifest has no file hashes")
    for relative, expected in hashes.items():
        path = _resource_path(root, str(relative))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"resource hash mismatch: {relative}")


@dataclass
class ProbabilityDecay:
    """Scalar form of state_t = max(raw_t, factor * state_(t-1))."""

    factor: float | None
    state: float = 0.0

    def step(self, probability: float) -> float:
        raw = _clip_probability(probability)
        if self.factor is None:
            return _float32_probability(raw)
        self.state = max(raw, self.factor * self.state)
        return _float32_probability(self.state)


@dataclass
class LogitDecay:
    """Scalar form of margin_t = max(raw_margin_t, margin_(t-1) - penalty)."""

    penalty: float
    state: float = -math.inf

    def step(self, probability: float) -> float:
        raw = _safe_logit(probability)
        self.state = max(raw, self.state - self.penalty)
        return _float32_probability(_logistic(self.state))


@dataclass
class LoadedFeatureVoter:
    """One XGBoost fold voter and its fold-specific feature projection."""

    model: Any
    columns: np.ndarray

    def predict(self, row: np.ndarray) -> float:
        values = np.ascontiguousarray(row[self.columns], dtype=np.float32)[None, :]
        prediction = self.model.inplace_predict(
            values,
            validate_features=False,
        )
        return _clip_probability(float(np.asarray(prediction).reshape(-1)[0]))


@dataclass
class LoadedFeatureComponent:
    name: str
    kind: str
    forest_bag: CompactForestBag | None = None
    voters: tuple[LoadedFeatureVoter, ...] = ()

    def predict(self, row: np.ndarray) -> float:
        if self.kind == "forest" and self.forest_bag is not None:
            probabilities = self.forest_bag.predict_probabilities(row)
        elif self.kind == "xgboost" and self.voters:
            probabilities = np.asarray(
                [voter.predict(row) for voter in self.voters], dtype=np.float64
            )
        else:  # pragma: no cover - bundle validation prevents this state.
            raise RuntimeError(f"feature component {self.name!r} is incomplete")
        return _bag_probabilities(probabilities)


@dataclass
class NativeDetectorEnsemble:
    """All detector voters traversed in one native LightGBM call."""

    feature_count: int
    voters_per_bank: int
    bank_count: int
    model: Any
    group_offsets: np.ndarray
    leaf_offsets: np.ndarray
    leaf_values: np.ndarray

    def __post_init__(self) -> None:
        tree_count = int(self.model.num_trees())
        group_count = self.bank_count * self.voters_per_bank
        if self.model.num_feature() != self.feature_count:
            raise ValueError("native detector feature count changed")
        if self.group_offsets.shape != (group_count + 1,):
            raise ValueError("native detector group offsets are malformed")
        if (
            self.group_offsets[0] != 0
            or self.group_offsets[-1] != tree_count
            or np.any(np.diff(self.group_offsets) <= 0)
        ):
            raise ValueError("native detector tree groups are malformed")
        if self.leaf_offsets.shape != (tree_count,):
            raise ValueError("native detector leaf offsets are malformed")
        if (
            self.leaf_values.ndim != 1
            or self.leaf_values.size == 0
            or self.leaf_offsets[0] != 0
            or np.any(np.diff(self.leaf_offsets) <= 0)
        ):
            raise ValueError("native detector leaf lookup is malformed")

    def predict_fold_margins(self, row: np.ndarray) -> np.ndarray:
        values = np.asarray(row, dtype=np.float32)
        if values.ndim != 1 or values.size != self.feature_count:
            raise ValueError("native detector received the wrong feature row")
        leaf_index = np.asarray(
            self.model.predict(values[None, :], pred_leaf=True, num_threads=1),
            dtype=np.int64,
        ).reshape(-1)
        if leaf_index.shape != self.leaf_offsets.shape:
            raise RuntimeError("native detector returned the wrong leaf count")
        locations = self.leaf_offsets + leaf_index
        if np.any(locations < 0) or np.any(locations >= self.leaf_values.size):
            raise RuntimeError("native detector returned an invalid leaf")
        margins = np.add.reduceat(
            self.leaf_values[locations], self.group_offsets[:-1]
        )
        return margins.reshape(self.bank_count, self.voters_per_bank)


@dataclass
class LoadedBundle:
    root: Path
    manifest: dict[str, Any]
    union_banks: tuple[str, ...]
    bank_order: dict[str, tuple[str, ...]]
    detector_ensemble: NativeDetectorEnsemble
    features: dict[str, LoadedFeatureComponent]
    linear_meta: CompactLinearLogistic | None
    lightgbm_meta: Any | None
    lightgbm_meta_iteration: int | None
    component_specs: tuple[dict[str, Any], ...]
    stream_detector_class: Any


def _name_columns(
    names: Sequence[str], name_to_column: dict[str, int], label: str
) -> np.ndarray:
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate feature names")
    try:
        return np.asarray([name_to_column[name] for name in names], dtype=np.int32)
    except KeyError as error:
        raise ValueError(f"{label} references unknown feature {error.args[0]!r}") from error


def _read_resource_manifest(
    model_directory_path: str, *, verify_hashes: bool
) -> tuple[Path, dict[str, Any]]:
    root = Path(model_directory_path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Submission 6 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2 or manifest.get("submission") != "submission_6":
        raise ValueError("model directory is not a Submission 6 resource bundle")
    if manifest.get("deployment_profile") != EXPECTED_DEPLOYMENT_PROFILE:
        raise ValueError("Submission 6 is not the timeout-safe deployment profile")
    if manifest.get("pretrained_models_bundled") is not True:
        raise ValueError("Submission 6 requires its frozen pretrained voters")
    if verify_hashes:
        _verify_hashes(root, manifest)
    return root, manifest


def _feature_runtime():
    try:
        from . import feature_pipeline as runtime
    except ImportError:
        import feature_pipeline as runtime
    return runtime


def _load_bundle(
    model_directory_path: str, *, verify_hashes: bool = False
) -> LoadedBundle:
    # These imports happen inside an inference worker. In particular, do not
    # initialize LightGBM/OpenMP in the force-first-train parent before Crunch
    # forks its worker pool.
    import lightgbm as lgb

    root, manifest = _read_resource_manifest(
        model_directory_path, verify_hashes=verify_hashes
    )
    feature_runtime = _feature_runtime()

    union_banks = tuple(str(value) for value in manifest["union_banks"])
    if len(set(union_banks)) != len(union_banks):
        raise ValueError("union bank contract contains duplicates")
    union_names = feature_runtime.feature_names(union_banks)
    name_to_column = {name: index for index, name in enumerate(union_names)}
    if len(name_to_column) != len(union_names):
        raise ValueError("detector code emitted duplicate union feature names")

    bank_order: dict[str, tuple[str, ...]] = {}
    for family, family_spec in manifest["detector_families"].items():
        order = tuple(str(value) for value in family_spec["bank_order"])
        if not order or len(order) != len(set(order)):
            raise ValueError(f"{family} detector order is empty or duplicated")
        bank_order[family] = order

    detector_spec = manifest["detector_ensemble"]
    declared_union_names = [str(value) for value in detector_spec["feature_names"]]
    if declared_union_names != union_names:
        raise ValueError("compact detector union feature contract changed")
    expected_bank_keys = [
        [family, bank]
        for family, order in bank_order.items()
        for bank in order
    ]
    if detector_spec["bank_keys"] != expected_bank_keys:
        raise ValueError("compact detector bank order changed")
    if detector_spec.get("kind") != "native_lightgbm_detector_ensemble":
        raise ValueError("Submission 6 requires the native detector runtime")
    detector_model = lgb.Booster(
        model_file=str(_resource_path(root, detector_spec["model_path"]))
    )
    with np.load(
        _resource_path(root, detector_spec["lookup_path"]), allow_pickle=False
    ) as payload:
        if str(payload["model_type"].item()) != "native_lightgbm_detector_ensemble":
            raise ValueError("native detector lookup has the wrong model type")
        detector_ensemble = NativeDetectorEnsemble(
            feature_count=int(payload["feature_count"].item()),
            voters_per_bank=int(payload["voters_per_bank"].item()),
            bank_count=int(payload["bank_count"].item()),
            model=detector_model,
            group_offsets=payload["group_offsets"].astype(np.int32, copy=True),
            leaf_offsets=payload["leaf_offsets"].astype(np.int32, copy=True),
            leaf_values=payload["leaf_values"].astype(np.float64, copy=True),
        )
    if (
        detector_ensemble.feature_count != len(union_names)
        or detector_ensemble.bank_count != len(expected_bank_keys)
        or detector_ensemble.voters_per_bank != 5
    ):
        raise ValueError("native detector ensemble dimensions changed")

    features: dict[str, LoadedFeatureComponent] = {}
    xgboost_booster_class: Any | None = None
    for component_spec in manifest["feature_components"]:
        name = str(component_spec["name"])
        kind = str(component_spec["kind"])
        if name in features:
            raise ValueError(f"duplicate feature component {name!r}")
        if kind == "forest":
            names = [str(value) for value in component_spec["feature_names"]]
            if names != union_names:
                raise ValueError(f"{name} forest union feature contract changed")
            forest_bag = load_compact_forest_bag(
                _resource_path(root, component_spec["path"])
            )
            if (
                forest_bag.feature_count != len(union_names)
                or forest_bag.voter_count != 5
            ):
                raise ValueError(f"{name} forest-bag dimensions changed")
            features[name] = LoadedFeatureComponent(
                name=name,
                kind=kind,
                forest_bag=forest_bag,
            )
            continue
        if kind == "xgboost":
            # Importing XGBoost here (inside an inference worker) is deliberate;
            # importing it from train() can initialize OpenMP before fork.
            if xgboost_booster_class is None:
                from xgboost import Booster

                xgboost_booster_class = Booster
            voters: list[LoadedFeatureVoter] = []
            for voter_spec in component_spec["voters"]:
                names = [str(value) for value in voter_spec["feature_names"]]
                columns = _name_columns(names, name_to_column, f"{name} voter")
                booster = xgboost_booster_class(params={"nthread": 1})
                booster.load_model(_resource_path(root, voter_spec["path"]))
                booster.set_param({"nthread": 1})
                if int(booster.num_features()) != columns.size:
                    raise ValueError(f"{name} XGBoost voter dimensions changed")
                voters.append(LoadedFeatureVoter(booster, columns))
            if len(voters) != 5:
                raise ValueError(f"{name} must contain five XGBoost voters")
            features[name] = LoadedFeatureComponent(
                name=name,
                kind=kind,
                voters=tuple(voters),
            )
            continue
        raise ValueError(f"unsupported feature component kind {kind!r}")

    meta_specs = manifest.get("meta_models", {})
    linear_meta: CompactLinearLogistic | None = None
    lightgbm_meta: Any | None = None
    lightgbm_meta_iteration: int | None = None
    if any(spec.get("kind") == "linear_meta" for spec in manifest["components"]):
        linear_spec = meta_specs.get("core_linear_l2")
        if not linear_spec or linear_spec.get("kind") != "linear_logistic":
            raise ValueError("core linear meta-model resource is missing")
        linear_meta = load_compact_linear(
            _resource_path(root, linear_spec["path"])
        )
        expected = len(bank_order["core"]) + 6
        if linear_meta.feature_count != expected:
            raise ValueError("core linear meta-model dimensions changed")
    if any(spec.get("kind") == "lightgbm_meta" for spec in manifest["components"]):
        lightgbm_spec = meta_specs.get("candidate_lightgbm_gbdt")
        if not lightgbm_spec or lightgbm_spec.get("kind") != "lightgbm":
            raise ValueError("candidate LightGBM meta-model resource is missing")
        lightgbm_meta = lgb.Booster(
            model_file=str(_resource_path(root, lightgbm_spec["path"]))
        )
        lightgbm_meta_iteration = int(lightgbm_spec["best_iteration"])
        expected = len(bank_order["candidate"]) + 6
        if lightgbm_meta.num_feature() != expected:
            raise ValueError("candidate LightGBM meta-model dimensions changed")

    component_specs = tuple(manifest["components"])
    component_names = [str(value["name"]) for value in component_specs]
    if len(component_names) != len(set(component_names)):
        raise ValueError("global component names are not unique")
    recipe_names = {manifest["global_recipe"]["start_component"]} | {
        step["component"] for step in manifest["global_recipe"]["steps"]
    }
    if recipe_names != set(component_names):
        raise ValueError("global recipe and loaded component set differ")
    return LoadedBundle(
        root=root,
        manifest=manifest,
        union_banks=union_banks,
        bank_order=bank_order,
        detector_ensemble=detector_ensemble,
        features=features,
        linear_meta=linear_meta,
        lightgbm_meta=lightgbm_meta,
        lightgbm_meta_iteration=lightgbm_meta_iteration,
        component_specs=component_specs,
        stream_detector_class=feature_runtime.CoreStreamDetector,
    )


def _decay_factor(specification: dict[str, Any]) -> float | None:
    kind = specification.get("kind")
    if kind == "raw":
        return None
    if kind != "probability_multiplicative_decayed_max":
        raise ValueError(f"Submission 6 does not support transform {kind!r}")
    factor = float(specification["decay_factor"])
    if not 0.0 <= factor <= 1.0:
        raise ValueError("probability decay factor must be in [0, 1]")
    return factor


def _causal_transform(specification: dict[str, Any]):
    kind = specification.get("kind")
    if kind in {"raw", "probability_multiplicative_decayed_max"}:
        return ProbabilityDecay(_decay_factor(specification))
    if kind == "logit_subtractive_decayed_max":
        penalty = float(specification["penalty"])
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError("logit decay penalty must be finite and non-negative")
        return LogitDecay(penalty)
    raise ValueError(f"Submission 6 does not support transform {kind!r}")


def _apply_recipe(values: dict[str, float], recipe: dict[str, Any]) -> float:
    current = _safe_logit(values[str(recipe["start_component"])])
    for step in recipe["steps"]:
        alpha = float(step["new_component_weight"])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("ensemble weight must be in [0, 1]")
        margin = _safe_logit(values[str(step["component"])])
        current = (1.0 - alpha) * current + alpha * margin
    return _logistic(current)


class SeriesPredictor:
    """All model and persistence state for exactly one streamed series."""

    def __init__(self, bundle: LoadedBundle, historical: Sequence[float]) -> None:
        self.bundle = bundle
        self.detector = bundle.stream_detector_class(
            np.asarray(historical, dtype=np.float64), bundle.union_banks
        )
        self.component_decay = {
            str(specification["name"]): _causal_transform(
                specification["transform"]
            )
            for specification in bundle.component_specs
        }
        self.global_decay = _causal_transform(bundle.manifest["global_transform"])

    def step(self, value: float) -> float:
        row = np.asarray(self.detector.step(float(value)), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)

        fold_margin = self.bundle.detector_ensemble.predict_fold_margins(
            row
        )
        bank_probability = _bag_margin_rows(fold_margin)
        family_probability: dict[str, np.ndarray] = {}
        offset = 0
        for family, order in self.bundle.bank_order.items():
            right = offset + len(order)
            family_probability[family] = bank_probability[offset:right]
            offset = right
        if offset != bank_probability.size:
            raise RuntimeError("native detector output cardinality changed")
        stack_features = {
            family: _stack_features(probabilities)
            for family, probabilities in family_probability.items()
        }
        raw_components: dict[str, float] = {}
        for specification in self.bundle.component_specs:
            name = str(specification["name"])
            kind = str(specification["kind"])
            if kind == "bank_logit_mean":
                family = str(specification["family"])
                raw = _logistic(
                    float(np.mean(_safe_logit_array(family_probability[family])))
                )
            elif kind == "feature_bag":
                raw = self.bundle.features[name].predict(row)
            elif kind == "linear_meta":
                family = str(specification["family"])
                if self.bundle.linear_meta is None:
                    raise RuntimeError("linear meta-model was not loaded")
                raw = self.bundle.linear_meta.predict_probability(
                    stack_features[family]
                )
            elif kind == "lightgbm_meta":
                family = str(specification["family"])
                if self.bundle.lightgbm_meta is None:
                    raise RuntimeError("LightGBM meta-model was not loaded")
                prediction = self.bundle.lightgbm_meta.predict(
                    stack_features[family][None, :],
                    num_iteration=self.bundle.lightgbm_meta_iteration,
                    num_threads=1,
                )
                raw = float(np.asarray(prediction).reshape(-1)[0])
            else:  # pragma: no cover - manifest validation catches this.
                raise ValueError(f"unknown component kind: {kind}")
            raw_components[name] = self.component_decay[name].step(
                _float32_probability(raw)
            )

        global_raw = _apply_recipe(
            raw_components, self.bundle.manifest["global_recipe"]
        )
        return self.global_decay.step(global_raw)


def _test_item_parts(item: tuple) -> tuple[Sequence[float], Iterable[float]]:
    if len(item) == 2:
        historical, online = item
    elif len(item) == 3:
        _, historical, online = item
    else:
        raise ValueError("test records must be (history, online) or (id, history, online)")
    return historical, online


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Validate and preserve the uploaded train-derived frozen resources.

    No competition test data or labels are read.  The five-fold voters were
    fitted only from the supplied public training folder during Model 6's
    leakage-controlled experiment, so a forced Crunch train call is a no-op
    after strict hash/model/feature-contract validation.
    """

    del datasets
    _read_resource_manifest(model_directory_path, verify_hashes=True)


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield readiness, then one probability immediately per revealed value."""

    bundle = _load_bundle(model_directory_path, verify_hashes=False)
    yield
    for item in datasets:
        historical, online = _test_item_parts(item)
        predictor = SeriesPredictor(bundle, historical)
        for value in online:
            yield predictor.step(float(value))


__all__ = ["infer", "train"]
