"""Immutable configuration for the train-only Model 7 experiment.

Model 7 deliberately has no holdout-evaluation command.  Model 6's outer
holdout has already been revealed, so every Model 7 decision and score uses
only the 6,666 series that were Model 6 development data.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL7_VERSION = "model7-tsauc-ranking-repeated-cv-v1"


@dataclass(frozen=True)
class Model7Config:
    """Choices that can alter a Model 7 fit, selection, or estimate."""

    project_root: Path
    target_tsroc: float = 0.64

    # These folds are rebuilt only inside Model 6's development population.
    repeat_seeds: tuple[int, ...] = (207_101, 207_503, 207_907)
    folds_per_repeat: int = 5
    screen_repeat: int = 0
    screen_folds: tuple[int, ...] = (0, 2)

    # Every score-fold ranking is fitted without its score and stopping folds.
    feature_family: str = "candidate"
    ranking_method: str = "fisher"
    feature_count: int = 256
    ranking_sample_rows: int = 60_000

    # Complete series, rather than arbitrary rows, are sampled for fitting.
    stop_rows: int = 300_000
    xgb_small_train_rows: int = 800_000
    large_train_rows: int = 1_200_000
    boosting_rounds: int = 650
    early_stopping_rounds: int = 65
    prediction_chunk_rows: int = 50_000

    # Pairwise XGBoost settings. Query groups are exact online time steps.
    pair_method: str = "mean"
    pairs_per_sample: int = 8
    xgb_learning_rate: float = 0.04
    xgb_max_depth: int = 3
    xgb_min_child_weight: float = 12.0
    xgb_subsample: float = 0.85
    xgb_colsample: float = 0.70
    xgb_reg_alpha: float = 0.20
    xgb_reg_lambda: float = 8.0
    xgb_max_bin: int = 127

    # GOSS is the stable pointwise baseline, now using three times more rows.
    goss_learning_rate: float = 0.035
    goss_num_leaves: int = 31
    goss_min_data_in_leaf: int = 250
    goss_feature_fraction: float = 0.65
    goss_lambda_l1: float = 0.20
    goss_lambda_l2: float = 4.0
    goss_top_rate: float = 0.20
    goss_other_rate: float = 0.10

    # Nested, low-capacity ensemble and causal absorbing-state filter grids.
    blend_weights: tuple[float, ...] = (
        0.0,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.0,
    )
    uncertainty_penalty: float = 0.5
    hmm_time_edges: tuple[int, ...] = (0, 32, 64, 128, 256, 512, 1_000)
    hmm_evidence_scales: tuple[float, ...] = (0.15, 0.25, 0.40, 0.60)
    hmm_hazard_scales: tuple[float, ...] = (0.50, 1.0, 2.0)
    bootstrap_replicates: int = 200
    bootstrap_seed: int = 270_031

    seed: int = 207_031
    num_threads: int = max(1, min(12, os.cpu_count() or 1))
    artifact_directory: str = ""

    @property
    def artifact_path(self) -> Path:
        return self.project_root / self.artifact_directory

    @property
    def model_names(self) -> tuple[str, ...]:
        return (
            "xgb_pair_d3_800k",
            "xgb_pair_d3_1200k",
            "goss_binary_1200k",
        )

    def protocol(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("project_root")
        return {
            "experiment_version": MODEL7_VERSION,
            "models": list(self.model_names),
            **values,
        }

    def identity_protocol(self) -> dict[str, Any]:
        values = self.protocol()
        values.pop("artifact_directory")
        return values

    def validate(self) -> None:
        if not 0.5 < self.target_tsroc < 1.0:
            raise ValueError("target_tsroc must be between 0.5 and 1")
        if len(self.repeat_seeds) < 2 or len(set(self.repeat_seeds)) != len(
            self.repeat_seeds
        ):
            raise ValueError("repeat_seeds must contain at least two unique seeds")
        if self.folds_per_repeat < 3:
            raise ValueError("at least three folds are required")
        if self.screen_repeat not in range(len(self.repeat_seeds)):
            raise ValueError("screen_repeat is invalid")
        if not self.screen_folds or any(
            fold not in range(self.folds_per_repeat) for fold in self.screen_folds
        ):
            raise ValueError("screen_folds are invalid")
        if self.feature_family != "candidate" or self.ranking_method != "fisher":
            raise ValueError("Model 7 currently supports candidate/Fisher only")
        if min(
            self.feature_count,
            self.ranking_sample_rows,
            self.stop_rows,
            self.xgb_small_train_rows,
            self.large_train_rows,
            self.boosting_rounds,
            self.early_stopping_rounds,
            self.prediction_chunk_rows,
            self.num_threads,
        ) < 1:
            raise ValueError("row, feature, iteration, and thread counts must be positive")
        if self.xgb_small_train_rows >= self.large_train_rows:
            raise ValueError("the small XGBoost cap must be below the large cap")
        if tuple(sorted(set(self.blend_weights))) != self.blend_weights:
            raise ValueError("blend_weights must be unique and increasing")
        if self.blend_weights[0] != 0.0 or self.blend_weights[-1] != 1.0:
            raise ValueError("blend_weights must include both endpoints")
        if len(self.hmm_time_edges) < 2 or tuple(sorted(self.hmm_time_edges)) != (
            self.hmm_time_edges
        ):
            raise ValueError("hmm_time_edges must be strictly increasing")
        if self.bootstrap_replicates < 20:
            raise ValueError("use at least 20 complete-series bootstrap replicates")


def _run_label(config: Model7Config) -> str:
    payload = json.dumps(
        config.identity_protocol(), sort_keys=True, separators=(",", ":")
    ).encode()
    return f"rankcv_{hashlib.sha256(payload).hexdigest()[:10]}"


def make_model7_config(project_root: str | Path) -> Model7Config:
    """Return the deterministic artifact location for the declared protocol."""

    root = Path(project_root).resolve()
    provisional = Model7Config(project_root=root)
    ready = Model7Config(
        **{
            **asdict(provisional),
            "project_root": root,
            "artifact_directory": f"models/model7/runs/{_run_label(provisional)}",
        }
    )
    ready.validate()
    return ready


CONFIG_HELP = {
    "repeat_seeds": "Three complete-series five-fold assignments; no Model 6 holdout series enters them.",
    "feature_count": "Top Fisher columns are recomputed using only each fold's fitting role.",
    "*_train_rows": "Approximate cap formed by sampling complete series, never arbitrary rows.",
    "pair_method": "XGBoost pair construction inside each exact time_online query group.",
    "uncertainty_penalty": "Selection maximizes mean TSAUC minus this multiplier times fold SE.",
    "hmm_*": "Nested causal absorbing-state filter grid; raw prediction remains an allowed endpoint.",
}
