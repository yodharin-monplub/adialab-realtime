"""User-editable, immutable-by-run configuration for Model 6.

Model 6 inherits Model 5's complete-series split and detector selection.  The
outer fold is never read during development.  Expensive searches use staged
promotion, and every setting below is written into the run manifest so a
changed protocol cannot accidentally reuse incompatible checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL6_VERSION = "model6-multimodel-tsroc-v2-probability-decay"


@dataclass(frozen=True)
class Model6Config:
    """All choices that can change a Model 6 selection or fitted model."""

    project_root: Path
    target_tsroc: float = 0.64
    source_detector_threshold: float = 0.55
    source_detector_se_multiplier: float = 0.5

    # Five train-derived ranking methods and the requested feature-count grid.
    ranking_methods: tuple[str, ...] = (
        "gain",
        "split",
        "consensus",
        "gain_corr",
        "fisher",
    )
    feature_counts: tuple[int, ...] = (32, 64, 128, 256, 512)
    include_all_features: bool = True
    ranking_sample_rows: int = 30_000
    correlation_sample_rows: int = 3_000
    correlation_pool_size: int = 768
    correlation_abs_threshold: float = 0.98

    # Stage 1: compare every method/count with the same LightGBM reference.
    feature_search_folds: tuple[int, ...] = (0, 2)
    feature_search_train_rows: int = 100_000
    feature_search_stop_rows: int = 60_000
    feature_search_rounds: int = 260
    feature_search_patience: int = 30
    promoted_feature_contracts: int = 6

    # Stage 2: compare a broad learner set on promoted feature contracts.
    model_types: tuple[str, ...] = (
        "lightgbm_gbdt",
        "lightgbm_dart",
        "lightgbm_goss",
        "lightgbm_rf",
        "xgboost",
        "catboost",
        "random_forest",
        "extra_trees",
        "linear_l2",
        "linear_l1",
        "linear_elasticnet",
    )
    model_search_folds: tuple[int, ...] = (1, 3)
    model_search_train_rows: int = 90_000
    model_search_stop_rows: int = 60_000
    model_search_rounds: int = 220
    model_search_patience: int = 25
    promoted_model_types: int = 4

    # Stage 3: tune only learners that survived Stage 2.
    tuning_folds: tuple[int, ...] = (0, 2, 4)
    tuning_train_rows: int = 160_000
    tuning_stop_rows: int = 90_000
    tuning_rounds: int = 420
    tuning_patience: int = 45
    finalists_per_family: int = 3

    # Stage 4: five-fold OOF predictions for finalists and deployment models.
    final_train_rows: int = 400_000
    final_stop_rows: int = 160_000
    final_rounds: int = 800
    final_patience: int = 80
    prediction_chunk_rows: int = 30_000

    # Detector OOF stacking and greedy logit-space ensembling.
    detector_stack_train_rows: int = 300_000
    ensemble_weight_step: float = 0.05
    ensemble_max_components: int = 8
    causal_penalties: tuple[float | None, ...] = (
        None,
        0.0,
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
    )

    # Probability-space absorbing-state hypothesis:
    #   state_t = max(raw_probability_t, factor * state_(t-1)).
    # The grid is dense near one because persistence changes fastest there for
    # long series.  Raw predictions are always tested as a separate baseline.
    probability_decay_factors: tuple[float, ...] = (
        1.0,
        0.9999,
        0.999,
        0.995,
        0.99,
        0.98,
        0.95,
        0.90,
        0.85,
        0.80,
        0.70,
        0.60,
        0.50,
    )

    # Fresh model/search seed paired with Model 5's new 10,000-series split.
    seed: int = 20_260_861
    num_threads: int = max(1, min(12, os.cpu_count() or 1))
    requested_device: str = "auto"
    artifact_directory: str = ""

    @property
    def artifact_path(self) -> Path:
        return self.project_root / self.artifact_directory

    def protocol(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("project_root")
        return {"experiment_version": MODEL6_VERSION, **values}

    def identity_protocol(self) -> dict[str, Any]:
        values = self.protocol()
        values.pop("artifact_directory")
        return values

    def validate(self) -> None:
        if not 0.5 < self.target_tsroc < 1.0:
            raise ValueError("target_tsroc must be between 0.5 and 1")
        if not 0.5 <= self.source_detector_threshold < 1.0:
            raise ValueError("source_detector_threshold must be in [0.5, 1)")
        if self.source_detector_se_multiplier < 0:
            raise ValueError("source_detector_se_multiplier must be non-negative")
        if len(set(self.ranking_methods)) != len(self.ranking_methods):
            raise ValueError("ranking_methods must be unique")
        supported = {"gain", "split", "consensus", "gain_corr", "fisher"}
        if set(self.ranking_methods) - supported:
            raise ValueError("an unsupported ranking method was configured")
        if tuple(sorted(set(self.feature_counts))) != self.feature_counts:
            raise ValueError("feature_counts must be unique and increasing")
        if any(value < 1 for value in self.feature_counts):
            raise ValueError("feature counts must be positive")
        for folds in (
            self.feature_search_folds,
            self.model_search_folds,
            self.tuning_folds,
        ):
            if not folds or any(fold not in range(5) for fold in folds):
                raise ValueError("search folds must be non-empty subsets of 0..4")
        if not 0.0 < self.correlation_abs_threshold < 1.0:
            raise ValueError("correlation_abs_threshold must be in (0, 1)")
        if not 0.0 < self.ensemble_weight_step <= 1.0:
            raise ValueError("ensemble_weight_step must be in (0, 1]")
        if not self.probability_decay_factors:
            raise ValueError("probability_decay_factors must not be empty")
        if len(set(self.probability_decay_factors)) != len(
            self.probability_decay_factors
        ):
            raise ValueError("probability_decay_factors must be unique")
        if any(
            not 0.0 <= value <= 1.0
            for value in self.probability_decay_factors
        ):
            raise ValueError("probability_decay_factors must be in [0, 1]")
        if tuple(sorted(self.probability_decay_factors, reverse=True)) != (
            self.probability_decay_factors
        ):
            raise ValueError("probability_decay_factors must be decreasing")
        if min(
            self.feature_search_train_rows,
            self.model_search_train_rows,
            self.tuning_train_rows,
            self.final_train_rows,
            self.prediction_chunk_rows,
            self.num_threads,
        ) < 1:
            raise ValueError("row caps, chunk size, and threads must be positive")
        if self.requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("requested_device must be auto, cpu, or cuda")


def _run_label(config: Model6Config) -> str:
    payload = json.dumps(
        config.identity_protocol(), sort_keys=True, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()[:10]
    return f"tsroc{str(config.target_tsroc).replace('.', 'p')}_{digest}"


def make_model6_config(project_root: str | Path) -> Model6Config:
    """Return a deterministic run directory for the current protocol."""

    root = Path(project_root).resolve()
    provisional = Model6Config(project_root=root)
    run = _run_label(provisional)
    ready = Model6Config(
        **{
            **asdict(provisional),
            "project_root": root,
            "artifact_directory": f"models/model6/runs/{run}",
        }
    )
    ready.validate()
    return ready


CONFIG_HELP = {
    "target_tsroc": "Goal only; it never changes labels or metric calculation.",
    "feature_counts": "Every ranking tests 32, 64, 128, 256, and 512 features.",
    "include_all_features": "Also test the complete selected detector family.",
    "*_folds": "Development folds used by that search stage; fold 0 here is not the outer holdout.",
    "*_train_rows": "Label-independent cap controlling runtime and memory.",
    "promoted_*": "Successive-halving width; larger values cost more but search more broadly.",
    "probability_decay_factors": (
        "Nested-CV grid for state=max(raw, factor*previous_state); 1 is a "
        "strict running maximum and 0.5 forgets quickly."
    ),
    "requested_device": "auto records and uses CUDA only when a working NVIDIA runtime exists.",
}
