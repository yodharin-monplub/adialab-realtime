"""Immutable configuration for Model 9's full-data scaling experiment."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL9_VERSION = "model9-full-data-crossfit-v2-row-sampling"


@dataclass(frozen=True)
class Model9Config:
    """Every choice that can affect a Model 9 fit, selection, or score."""

    project_root: Path
    target_tsroc: float = 0.64
    folds: int = 5

    # These contracts were fixed before Model 9 scoring. Per-fold Fisher/split
    # columns come from Model 6 rankings that excluded that score fold.
    architectures: tuple[str, ...] = (
        "candidate_all_gbdt",
        "candidate_split512_gbdt",
        "core_fisher512_gbdt",
        "joint_fisher768_gbdt",
    )
    scale_train_rows: tuple[int, ...] = (400_000, 1_200_000)
    scale_stop_rows: int = 200_000
    scale_rounds: int = 1_000
    scale_patience: int = 100

    # Only architectures selected without a held fold are promoted. The cap is
    # near the complete three-fold fitting role while staying below 30 GiB RAM.
    promoted_train_rows: int = 2_000_000
    promoted_stop_rows: int = 250_000
    maximum_promoted_architectures: int = 4

    # A shallow cross-fitted stack learns interactions among OOF component
    # logits and online age. Its inputs never contain in-fold base predictions.
    meta_train_rows: int = 2_000_000
    meta_stop_rows: int = 250_000
    meta_rounds: int = 450
    meta_patience: int = 50

    blend_weights: tuple[float, ...] = tuple(index / 20.0 for index in range(21))
    blend_tuning_series_per_fold: int = 600
    uncertainty_penalty: float = 0.5
    prediction_chunk_rows: int = 30_000
    bootstrap_replicates: int = 200
    bootstrap_seed: int = 290_091
    seed: int = 29_009_001
    num_threads: int = max(1, min(12, os.cpu_count() or 1))

    source_model6_directory: str = "models/model6/runs/tsroc0p64_909b8f5c6c"
    source_baseline_prediction: str = "ensemble/global/nested_oof.f32.npy"
    artifact_directory: str = ""

    @property
    def artifact_path(self) -> Path:
        return self.project_root / self.artifact_directory

    @property
    def source_model6_path(self) -> Path:
        return self.project_root / self.source_model6_directory

    def protocol(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("project_root")
        return {"experiment_version": MODEL9_VERSION, **values}

    def identity_protocol(self) -> dict[str, Any]:
        values = self.protocol()
        values.pop("artifact_directory")
        return values

    def validate(self) -> None:
        if not 0.5 < self.target_tsroc < 1.0:
            raise ValueError("target_tsroc must be between 0.5 and 1")
        if self.folds != 5:
            raise ValueError("Model 9 must align with Model 6's five folds")
        expected = (
            "candidate_all_gbdt",
            "candidate_split512_gbdt",
            "core_fisher512_gbdt",
            "joint_fisher768_gbdt",
        )
        if self.architectures != expected:
            raise ValueError("Model 9's architecture contract is fixed")
        if tuple(sorted(set(self.scale_train_rows))) != self.scale_train_rows:
            raise ValueError("scale_train_rows must be unique and increasing")
        if self.scale_train_rows[0] >= self.scale_train_rows[-1]:
            raise ValueError("Model 9 requires at least two increasing row scales")
        if tuple(sorted(set(self.blend_weights))) != self.blend_weights:
            raise ValueError("blend weights must be unique and increasing")
        if self.blend_weights[0] != 0.0 or self.blend_weights[-1] != 1.0:
            raise ValueError("blend grid must contain both endpoints")
        if self.uncertainty_penalty < 0:
            raise ValueError("uncertainty penalty must be non-negative")
        positive = (
            *self.scale_train_rows,
            self.scale_stop_rows,
            self.scale_rounds,
            self.scale_patience,
            self.promoted_train_rows,
            self.promoted_stop_rows,
            self.maximum_promoted_architectures,
            self.blend_tuning_series_per_fold,
            self.meta_train_rows,
            self.meta_stop_rows,
            self.meta_rounds,
            self.meta_patience,
            self.prediction_chunk_rows,
            self.bootstrap_replicates,
            self.num_threads,
        )
        if min(positive) < 1:
            raise ValueError("row caps, rounds, workers, and replicates must be positive")


def _run_label(config: Model9Config) -> str:
    payload = json.dumps(
        config.identity_protocol(), sort_keys=True, separators=(",", ":")
    ).encode()
    return f"fulldata_{hashlib.sha256(payload).hexdigest()[:10]}"


def make_model9_config(project_root: str | Path) -> Model9Config:
    """Return the deterministic run directory for the fixed Model 9 protocol."""

    root = Path(project_root).resolve()
    provisional = Model9Config(project_root=root)
    ready = Model9Config(
        **{
            **asdict(provisional),
            "project_root": root,
            "artifact_directory": f"models/model9/runs/{_run_label(provisional)}",
        }
    )
    ready.validate()
    return ready


CONFIG_HELP = {
    "architectures": "Four fixed LightGBM contracts; no outer-holdout selection.",
    "scale_train_rows": (
        "Nested label-independent row samples across every eligible fit-role "
        "series: 400k versus 1.2M."
    ),
    "promoted_train_rows": (
        "Two-million-row nested, label-independent sample within fit series; "
        "chosen to keep candidate-all below the 30 GiB machine limit."
    ),
    "*_stop_rows": (
        "Nested row sample inside a complete-series-disjoint early-stopping role."
    ),
    "uncertainty_penalty": "Selection uses mean fold TSAUC minus 0.5 times fold SE.",
    "source_baseline_prediction": "Cross-fitted Model 6 OOF only; holdout is prohibited.",
}
