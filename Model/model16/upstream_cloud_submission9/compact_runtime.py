"""Small, pickle-free runtimes used by Submission 6.

The experiment fitted LightGBM detector banks, sklearn forests, and an SGD
logistic meta-model. The bundle builder exports only the numerical state needed
for prediction, so Crunch inference does not need sklearn or joblib. Trees are
jointly traversed to reduce per-yield overhead while preserving fitted precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _scalar(payload: object, name: str, cast):
    value = np.asarray(payload[name])
    if value.ndim != 0:
        raise ValueError(f"compact field {name!r} must be scalar")
    return cast(value.item())


@dataclass
class CompactForest:
    """Vectorized one-row traversal for RandomForest and ExtraTrees voters."""

    feature_count: int
    max_depth: int
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    positive_probability: np.ndarray

    def __post_init__(self) -> None:
        shape = self.feature.shape
        if len(shape) != 2 or shape[0] == 0:
            raise ValueError("compact forest must contain a 2-D tree matrix")
        for array in (
            self.children_left,
            self.children_right,
            self.threshold,
            self.positive_probability,
        ):
            if array.shape != shape:
                raise ValueError("compact forest node arrays are not aligned")
        self._tree_index = np.arange(shape[0], dtype=np.int32)

    def predict_probability(self, row: np.ndarray) -> float:
        values = np.asarray(row, dtype=np.float32)
        if values.ndim != 1 or values.size != self.feature_count:
            raise ValueError("compact forest received the wrong feature count")
        nodes = np.zeros(self._tree_index.size, dtype=np.int32)
        for _ in range(self.max_depth + 1):
            split_feature = self.feature[self._tree_index, nodes]
            active = split_feature >= 0
            if not np.any(active):
                break
            trees = self._tree_index[active]
            active_nodes = nodes[active]
            go_left = (
                values[split_feature[active]]
                <= self.threshold[trees, active_nodes]
            )
            nodes[active] = np.where(
                go_left,
                self.children_left[trees, active_nodes],
                self.children_right[trees, active_nodes],
            )
        if np.any(self.feature[self._tree_index, nodes] >= 0):
            raise RuntimeError("compact forest traversal exceeded max_depth")
        return float(self.positive_probability[self._tree_index, nodes].mean())


@dataclass
class CompactForestBag:
    """Five sklearn forest voters traversed together on the union feature row."""

    feature_count: int
    max_depth: int
    voter_count: int
    roots: np.ndarray
    group_offsets: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    positive_probability: np.ndarray

    def __post_init__(self) -> None:
        node_count = int(self.feature.size)
        if node_count == 0 or self.feature.ndim != 1:
            raise ValueError("compact forest bag has no nodes")
        for array in (
            self.children_left,
            self.children_right,
            self.threshold,
            self.positive_probability,
        ):
            if array.shape != self.feature.shape:
                raise ValueError("compact forest-bag node arrays are not aligned")
        if self.roots.ndim != 1 or self.roots.size == 0:
            raise ValueError("compact forest bag has no trees")
        if self.group_offsets.shape != (self.voter_count + 1,):
            raise ValueError("compact forest-bag offsets are malformed")
        if (
            self.group_offsets[0] != 0
            or self.group_offsets[-1] != self.roots.size
            or np.any(np.diff(self.group_offsets) <= 0)
        ):
            raise ValueError("compact forest voters must contain trees")
        if np.any(self.roots < 0) or np.any(self.roots >= node_count):
            raise ValueError("compact forest-bag root is out of bounds")
        split = self.feature >= 0
        if np.any(self.feature[split] >= self.feature_count):
            raise ValueError("compact forest-bag feature is out of bounds")

    def predict_probabilities(self, row: np.ndarray) -> np.ndarray:
        values = np.asarray(row, dtype=np.float32)
        if values.ndim != 1 or values.size != self.feature_count:
            raise ValueError("compact forest bag received the wrong feature count")
        nodes = self.roots.copy()
        for _ in range(self.max_depth + 1):
            split_feature = self.feature[nodes]
            active = split_feature >= 0
            if not np.any(active):
                break
            active_nodes = nodes[active]
            go_left = (
                values[split_feature[active]] <= self.threshold[active_nodes]
            )
            nodes[active] = np.where(
                go_left,
                self.children_left[active_nodes],
                self.children_right[active_nodes],
            )
        if np.any(self.feature[nodes] >= 0):
            raise RuntimeError("compact forest-bag traversal exceeded max_depth")
        leaves = self.positive_probability[nodes]
        totals = np.add.reduceat(leaves, self.group_offsets[:-1])
        counts = np.diff(self.group_offsets)
        return totals / counts


@dataclass
class CompactLightGBMEnsemble:
    """All detector-bank LightGBM voters in one vectorized tree traversal.

    Trees are ordered by bank, then score-excluded voter, then boosting
    iteration. ``group_offsets`` partitions their leaf outputs into the five
    raw margins belonging to each detector bank. The source models use only
    finite numerical ``<=`` splits and the binary sigmoid objective; the
    exporter rejects any model outside that deliberately narrow contract.
    """

    feature_count: int
    max_depth: int
    voters_per_bank: int
    bank_count: int
    roots: np.ndarray
    group_offsets: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    leaf_value: np.ndarray

    def __post_init__(self) -> None:
        node_count = int(self.feature.size)
        if node_count == 0 or self.feature.ndim != 1:
            raise ValueError("compact LightGBM ensemble has no nodes")
        for array in (
            self.children_left,
            self.children_right,
            self.threshold,
            self.leaf_value,
        ):
            if array.shape != self.feature.shape:
                raise ValueError("compact LightGBM node arrays are not aligned")
        if self.roots.ndim != 1 or self.roots.size == 0:
            raise ValueError("compact LightGBM ensemble has no tree roots")
        expected_groups = self.bank_count * self.voters_per_bank
        if self.group_offsets.shape != (expected_groups + 1,):
            raise ValueError("compact LightGBM group offsets are malformed")
        if (
            self.group_offsets[0] != 0
            or self.group_offsets[-1] != self.roots.size
            or np.any(np.diff(self.group_offsets) <= 0)
        ):
            raise ValueError("compact LightGBM tree groups must be non-empty")
        if np.any(self.roots < 0) or np.any(self.roots >= node_count):
            raise ValueError("compact LightGBM tree root is out of bounds")
        split = self.feature >= 0
        if np.any(self.feature[split] >= self.feature_count):
            raise ValueError("compact LightGBM split feature is out of bounds")
        if np.any(self.children_left[split] < 0) or np.any(
            self.children_right[split] < 0
        ):
            raise ValueError("compact LightGBM split has a missing child")

    def predict_fold_probabilities(self, row: np.ndarray) -> np.ndarray:
        """Return ``(bank, voter)`` probabilities for one union-feature row."""

        values = np.asarray(row, dtype=np.float32)
        if values.ndim != 1 or values.size != self.feature_count:
            raise ValueError("compact LightGBM ensemble received the wrong row")
        nodes = self.roots.copy()
        for _ in range(self.max_depth + 1):
            split_feature = self.feature[nodes]
            active = split_feature >= 0
            if not np.any(active):
                break
            active_nodes = nodes[active]
            go_left = (
                values[split_feature[active]] <= self.threshold[active_nodes]
            )
            nodes[active] = np.where(
                go_left,
                self.children_left[active_nodes],
                self.children_right[active_nodes],
            )
        if np.any(self.feature[nodes] >= 0):
            raise RuntimeError("compact LightGBM traversal exceeded max_depth")

        leaves = self.leaf_value[nodes]
        margins = np.add.reduceat(leaves, self.group_offsets[:-1])
        # Stable scalar sigmoid, vectorized. The caller deliberately converts
        # these probabilities back to float32 logits before fold averaging,
        # exactly like the frozen Model 6 holdout path.
        probability = np.empty_like(margins)
        positive = margins >= 0.0
        probability[positive] = 1.0 / (1.0 + np.exp(-margins[positive]))
        exponential = np.exp(margins[~positive])
        probability[~positive] = exponential / (1.0 + exponential)
        return probability.reshape(self.bank_count, self.voters_per_bank)


@dataclass
class CompactLinearLogistic:
    """StandardScaler plus binary linear-logistic decision function."""

    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    clip: float

    @property
    def feature_count(self) -> int:
        return int(self.coefficient.size)

    def predict_probability(self, row: np.ndarray) -> float:
        values = np.asarray(row, dtype=np.float64)
        if values.ndim != 1 or values.size != self.feature_count:
            raise ValueError("compact linear model received the wrong feature count")
        # StandardScaler.transform preserves a float32 input and performs the
        # subtract/divide operations in place. Reproduce those two rounding
        # points because the frozen holdout path used a float32 stack matrix.
        transformed = values.astype(np.float32, copy=True)
        transformed -= self.mean
        transformed /= self.scale
        np.clip(transformed, -self.clip, self.clip, out=transformed)
        margin = float(np.dot(self.coefficient, transformed) + self.intercept)
        if margin >= 0.0:
            return 1.0 / (1.0 + np.exp(-margin))
        exponential = float(np.exp(margin))
        return exponential / (1.0 + exponential)


def load_compact_forest(path: str | Path) -> CompactForest:
    """Load and validate one compressed forest voter."""

    with np.load(path, allow_pickle=False) as payload:
        if _scalar(payload, "format_version", int) != 1:
            raise ValueError("unsupported compact forest version")
        if _scalar(payload, "model_type", str) != "forest":
            raise ValueError("compact resource is not a forest")
        return CompactForest(
            feature_count=_scalar(payload, "feature_count", int),
            max_depth=_scalar(payload, "max_depth", int),
            children_left=np.asarray(payload["children_left"], dtype=np.int32),
            children_right=np.asarray(payload["children_right"], dtype=np.int32),
            feature=np.asarray(payload["feature"], dtype=np.int32),
            threshold=np.asarray(payload["threshold"], dtype=np.float64),
            positive_probability=np.asarray(
                payload["positive_probability"], dtype=np.float64
            ),
        )


def load_compact_forest_bag(path: str | Path) -> CompactForestBag:
    """Load one jointly traversed five-voter forest component."""

    with np.load(path, allow_pickle=False) as payload:
        if _scalar(payload, "format_version", int) != 1:
            raise ValueError("unsupported compact forest-bag version")
        if _scalar(payload, "model_type", str) != "forest_bag":
            raise ValueError("compact resource is not a forest bag")
        return CompactForestBag(
            feature_count=_scalar(payload, "feature_count", int),
            max_depth=_scalar(payload, "max_depth", int),
            voter_count=_scalar(payload, "voter_count", int),
            roots=np.asarray(payload["roots"], dtype=np.int32),
            group_offsets=np.asarray(payload["group_offsets"], dtype=np.int32),
            children_left=np.asarray(payload["children_left"], dtype=np.int32),
            children_right=np.asarray(payload["children_right"], dtype=np.int32),
            feature=np.asarray(payload["feature"], dtype=np.int32),
            threshold=np.asarray(payload["threshold"], dtype=np.float64),
            positive_probability=np.asarray(
                payload["positive_probability"], dtype=np.float64
            ),
        )


def load_compact_lightgbm_ensemble(path: str | Path) -> CompactLightGBMEnsemble:
    """Load the merged detector-bank ensemble without native Booster objects."""

    with np.load(path, allow_pickle=False) as payload:
        if _scalar(payload, "format_version", int) != 1:
            raise ValueError("unsupported compact LightGBM version")
        if _scalar(payload, "model_type", str) != "lightgbm_detector_ensemble":
            raise ValueError("compact resource is not a LightGBM ensemble")
        return CompactLightGBMEnsemble(
            feature_count=_scalar(payload, "feature_count", int),
            max_depth=_scalar(payload, "max_depth", int),
            voters_per_bank=_scalar(payload, "voters_per_bank", int),
            bank_count=_scalar(payload, "bank_count", int),
            roots=np.asarray(payload["roots"], dtype=np.int32),
            group_offsets=np.asarray(payload["group_offsets"], dtype=np.int32),
            children_left=np.asarray(payload["children_left"], dtype=np.int32),
            children_right=np.asarray(payload["children_right"], dtype=np.int32),
            feature=np.asarray(payload["feature"], dtype=np.int32),
            threshold=np.asarray(payload["threshold"], dtype=np.float64),
            leaf_value=np.asarray(payload["leaf_value"], dtype=np.float64),
        )


def load_compact_linear(path: str | Path) -> CompactLinearLogistic:
    """Load and validate the selected detector-stack linear model."""

    with np.load(path, allow_pickle=False) as payload:
        if _scalar(payload, "format_version", int) != 1:
            raise ValueError("unsupported compact linear version")
        if _scalar(payload, "model_type", str) != "linear_logistic":
            raise ValueError("compact resource is not linear logistic")
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        coefficient = np.asarray(payload["coefficient"], dtype=np.float64)
        if mean.ndim != 1 or scale.shape != mean.shape or coefficient.shape != mean.shape:
            raise ValueError("compact linear arrays are not aligned")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("compact linear scale must be finite and positive")
        return CompactLinearLogistic(
            mean=mean,
            scale=scale,
            coefficient=coefficient,
            intercept=_scalar(payload, "intercept", float),
            clip=_scalar(payload, "clip", float),
        )


__all__ = [
    "CompactForest",
    "CompactForestBag",
    "CompactLightGBMEnsemble",
    "CompactLinearLogistic",
    "load_compact_forest",
    "load_compact_forest_bag",
    "load_compact_lightgbm_ensemble",
    "load_compact_linear",
]
