"""Immutable configuration for the Model 8 detector-expansion experiment."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL8_VERSION = "model8-new-detectors-nested-v1"


@dataclass(frozen=True)
class Model8Config:
    """Every choice affecting Model 8 fitting, selection, or reporting."""

    project_root: Path
    target_tsroc: float = 0.64
    detector_banks: tuple[str, ...] = ("d61", "d62", "d63", "d64", "d65", "d66")
    folds: int = 5

    # Each score fold is excluded. The next fold supplies early stopping and
    # the other three folds supply complete-series training rows.
    screen_train_rows: int = 300_000
    screen_stop_rows: int = 120_000
    screen_rounds: int = 500
    screen_patience: int = 50
    combined_train_rows: int = 600_000
    combined_stop_rows: int = 180_000
    combined_rounds: int = 800
    combined_patience: int = 80
    prediction_chunk_rows: int = 50_000

    uncertainty_penalty: float = 0.5
    detector_chance_threshold: float = 0.5
    blend_weights: tuple[float, ...] = tuple(index / 20.0 for index in range(21))
    greedy_max_new_banks: int = 3
    greedy_min_robust_gain: float = 0.0

    bootstrap_replicates: int = 200
    bootstrap_seed: int = 280_031
    seed: int = 218_031
    num_threads: int = max(1, min(12, os.cpu_count() or 1))
    cache_workers: int = max(1, min(8, os.cpu_count() or 1))

    source_model6_directory: str = "models/model6/runs/tsroc0p64_909b8f5c6c"
    source_baseline_prediction: str = "ensemble/candidate/nested_oof.f32.npy"
    cache_directory: str = "models/model8/cache_v1"
    cache_name: str = "model8_new_features"
    artifact_directory: str = ""

    @property
    def artifact_path(self) -> Path:
        return self.project_root / self.artifact_directory

    @property
    def cache_path(self) -> Path:
        return self.project_root / self.cache_directory

    @property
    def source_model6_path(self) -> Path:
        return self.project_root / self.source_model6_directory

    def protocol(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("project_root")
        return {"experiment_version": MODEL8_VERSION, **values}

    def identity_protocol(self) -> dict[str, Any]:
        values = self.protocol()
        values.pop("artifact_directory")
        return values

    def validate(self) -> None:
        if not 0.5 < self.target_tsroc < 1.0:
            raise ValueError("target_tsroc must be between 0.5 and 1")
        if self.detector_banks != ("d61", "d62", "d63", "d64", "d65", "d66"):
            raise ValueError("Model 8's six-bank contract is fixed")
        if self.folds != 5:
            raise ValueError("Model 8 must align with Model 6's five OOF folds")
        if tuple(sorted(set(self.blend_weights))) != self.blend_weights:
            raise ValueError("blend_weights must be unique and increasing")
        if self.blend_weights[0] != 0.0 or self.blend_weights[-1] != 1.0:
            raise ValueError("blend grid must contain both endpoints")
        if self.uncertainty_penalty < 0 or self.greedy_min_robust_gain < 0:
            raise ValueError("robustness penalties must be non-negative")
        positive = (
            self.screen_train_rows,
            self.screen_stop_rows,
            self.screen_rounds,
            self.screen_patience,
            self.combined_train_rows,
            self.combined_stop_rows,
            self.combined_rounds,
            self.combined_patience,
            self.prediction_chunk_rows,
            self.greedy_max_new_banks,
            self.bootstrap_replicates,
            self.num_threads,
            self.cache_workers,
        )
        if min(positive) < 1:
            raise ValueError("row caps, rounds, workers, and replicate counts must be positive")


def _run_label(config: Model8Config) -> str:
    payload = json.dumps(
        config.identity_protocol(), sort_keys=True, separators=(",", ":")
    ).encode()
    return f"newdet_{hashlib.sha256(payload).hexdigest()[:10]}"


def make_model8_config(project_root: str | Path) -> Model8Config:
    """Construct the deterministic output directory for the fixed protocol."""

    root = Path(project_root).resolve()
    provisional = Model8Config(project_root=root)
    ready = Model8Config(
        **{
            **asdict(provisional),
            "project_root": root,
            "artifact_directory": f"models/model8/runs/{_run_label(provisional)}",
        }
    )
    ready.validate()
    return ready


CONFIG_HELP = {
    "detector_banks": "The six new causal candidate banks; this contract is fixed.",
    "*_train_rows": "Label-independent complete-series row cap for fitting.",
    "*_stop_rows": "Disjoint complete-series cap used only for early stopping.",
    "uncertainty_penalty": "Selection maximizes mean fold TSAUC minus this multiple of fold SE.",
    "blend_weights": "Nested logit-space baseline/new-model blend grid, including both endpoints.",
    "source_baseline_prediction": "Cross-fitted Model 6 candidate OOF prediction; holdout predictions are prohibited.",
}
