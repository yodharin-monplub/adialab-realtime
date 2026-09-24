"""Frozen configuration for the Model 10 causal post-processing study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from models.model10.causal_transforms import TransformSpec


MODEL10_VERSION = "model10-causal-postprocess-v3"


@dataclass(frozen=True)
class Model10Config:
    """Every predeclared setting that can affect Model 10 selection or score."""

    project_root: Path
    folds: int = 5
    uncertainty_penalty: float = 0.5
    minimum_mean_gain: float = 0.00025
    minimum_robust_gain: float = 0.00010
    minimum_outer_fold_wins: int = 4
    calibration_subfolds: int = 5

    # s_t = logit(p_t) + weight * (logit(p_t) - EMA_t).
    ema_half_lives: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    ema_trend_weights: tuple[float, ...] = (
        -0.35,
        -0.20,
        -0.10,
        -0.05,
        0.05,
        0.10,
        0.20,
        0.35,
    )

    # s_t = logit(p_t) + weight * (rolling_max_t - logit(p_t)).
    rolling_max_windows: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    rolling_max_weights: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)

    # Blend the current logit toward a probability peak that decays each row.
    # Weight < 1 makes this deliberately softer than Model 6's hard replacement.
    decayed_peak_factors: tuple[float, ...] = (
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
        0.98,
        0.99,
        0.995,
        0.999,
        1.0,
    )
    decayed_peak_weights: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35)

    # The innovation is evaluated against the EW state at t-1, then clipped.
    innovation_half_lives: tuple[float, ...] = (16.0, 64.0, 256.0)
    innovation_weights: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.20)
    innovation_minimum_scale: float = 0.25
    innovation_clip: float = 6.0

    bootstrap_replicates: int = 1_000
    bootstrap_seed: int = 310_010
    seed: int = 31_001_001
    source_model6_directory: str = "models/model6/runs/tsroc0p64_909b8f5c6c"
    source_model9_directory: str = "models/model9/runs/fulldata_819a001db6"
    artifact_directory: str = ""

    @property
    def artifact_path(self) -> Path:
        """Return the configuration-derived Model 10 output directory."""

        return self.project_root / self.artifact_directory

    @property
    def source_model6_path(self) -> Path:
        """Return the read-only incumbent Model 6 directory."""

        return self.project_root / self.source_model6_directory

    @property
    def source_model9_path(self) -> Path:
        """Return the read-only Model 9 exploratory-stack directory."""

        return self.project_root / self.source_model9_directory

    def transform_specs(self) -> tuple[TransformSpec, ...]:
        """Return the complete ordered grid, with raw as the first fallback."""

        candidates: list[TransformSpec] = [TransformSpec(kind="raw")]
        candidates.extend(
            TransformSpec(
                kind="logit_ema_trend",
                half_life=half_life,
                correction_weight=weight,
            )
            for half_life in self.ema_half_lives
            for weight in self.ema_trend_weights
        )
        candidates.extend(
            TransformSpec(
                kind="rolling_max_premium",
                window=window,
                correction_weight=weight,
            )
            for window in self.rolling_max_windows
            for weight in self.rolling_max_weights
        )
        candidates.extend(
            TransformSpec(
                kind="decayed_peak_premium",
                decay_factor=factor,
                correction_weight=weight,
            )
            for factor in self.decayed_peak_factors
            for weight in self.decayed_peak_weights
        )
        candidates.extend(
            TransformSpec(
                kind="prior_ew_innovation",
                half_life=half_life,
                correction_weight=weight,
                minimum_scale=self.innovation_minimum_scale,
                innovation_clip=self.innovation_clip,
            )
            for half_life in self.innovation_half_lives
            for weight in self.innovation_weights
        )
        specs = tuple(candidates)
        for spec in specs:
            spec.validate()
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("Model 10 transform grid contains duplicate names")
        return specs

    def protocol(self) -> dict[str, Any]:
        """Return the JSON-safe experiment contract recorded with each run."""

        values = asdict(self)
        values.pop("project_root")
        return {
            "experiment_version": MODEL10_VERSION,
            **values,
            "transform_specs": [spec.to_dict() for spec in self.transform_specs()],
        }

    def identity_protocol(self) -> dict[str, Any]:
        """Return the run identity without its derived output path."""

        values = self.protocol()
        values.pop("artifact_directory")
        return values

    def validate(self) -> None:
        """Validate folds, selection safeguards, grids, and uncertainty settings."""

        if self.folds != 5:
            raise ValueError("Model 10 must align with Model 6's five folds")
        if self.calibration_subfolds != 5:
            raise ValueError("Model 10 requires five nested post-process subfolds")
        if not 0.0 <= self.uncertainty_penalty <= 2.0:
            raise ValueError("uncertainty_penalty must be in [0, 2]")
        if not 0.0 <= self.minimum_mean_gain < 0.1:
            raise ValueError("minimum_mean_gain must be in [0, 0.1)")
        if not 0.0 <= self.minimum_robust_gain < 0.1:
            raise ValueError("minimum_robust_gain must be in [0, 0.1)")
        if not 1 <= self.minimum_outer_fold_wins <= self.folds:
            raise ValueError("minimum_outer_fold_wins must be in [1, folds]")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")

        increasing_grids = (
            ("ema_half_lives", self.ema_half_lives),
            ("rolling_max_windows", self.rolling_max_windows),
            ("rolling_max_weights", self.rolling_max_weights),
            ("decayed_peak_factors", self.decayed_peak_factors),
            ("decayed_peak_weights", self.decayed_peak_weights),
            ("innovation_half_lives", self.innovation_half_lives),
            ("innovation_weights", self.innovation_weights),
        )
        for label, grid in increasing_grids:
            if not grid or tuple(sorted(set(grid))) != grid:
                raise ValueError(f"{label} must be non-empty, unique, and increasing")
        if not self.ema_trend_weights:
            raise ValueError("ema_trend_weights must not be empty")
        if tuple(sorted(set(self.ema_trend_weights))) != self.ema_trend_weights:
            raise ValueError("ema_trend_weights must be unique and increasing")
        if 0.0 in self.ema_trend_weights:
            raise ValueError("raw already represents a zero EMA correction")
        if min(self.ema_half_lives + self.innovation_half_lives) <= 0.0:
            raise ValueError("half-lives must be positive")
        if min(self.rolling_max_windows) < 2:
            raise ValueError("rolling windows must be at least two")
        positive_weights = (
            self.rolling_max_weights
            + self.decayed_peak_weights
            + self.innovation_weights
        )
        if min(positive_weights) <= 0.0:
            raise ValueError("rolling, peak, and innovation weights must be positive")
        if max(positive_weights) > 1.0:
            raise ValueError("correction weights must not exceed one")
        if not 0.0 < min(self.decayed_peak_factors):
            raise ValueError("decayed_peak_factors must be positive")
        if max(self.decayed_peak_factors) > 1.0:
            raise ValueError("decayed_peak_factors must not exceed one")
        if self.innovation_minimum_scale <= 0.0:
            raise ValueError("innovation_minimum_scale must be positive")
        if self.innovation_clip <= 0.0:
            raise ValueError("innovation_clip must be positive")
        self.transform_specs()


def _run_label(config: Model10Config) -> str:
    payload = json.dumps(
        config.identity_protocol(), sort_keys=True, separators=(",", ":")
    ).encode()
    return f"postprocess_{hashlib.sha256(payload).hexdigest()[:10]}"


def make_model10_config(project_root: str | Path) -> Model10Config:
    """Return the deterministic run directory for the frozen Model 10 grid."""

    root = Path(project_root).resolve()
    provisional = Model10Config(project_root=root)
    ready = Model10Config(
        **{
            **asdict(provisional),
            "project_root": root,
            "artifact_directory": f"models/model10/runs/{_run_label(provisional)}",
        }
    )
    ready.validate()
    return ready


CONFIG_HELP = {
    "ema_*": "Small signed trend corrections around the current raw logit.",
    "rolling_max_*": "Finite-window maximum premium; never a permanent run-max.",
    "decayed_peak_*": "Soft blend toward a decaying probability peak; one scalar of streaming state.",
    "innovation_*": "Prior-only EW self-normalization with a clipped correction.",
    "uncertainty_penalty": "Inner selection uses paired mean gain minus 0.5 times its SE.",
    "minimum_mean_gain": "Raw wins unless a correction improves mean TSAUC by 0.00025.",
    "minimum_robust_gain": "Require mean gain minus 0.5 SE to exceed 0.00010.",
    "minimum_outer_fold_wins": "A deployable correction must improve at least four of five folds.",
    "calibration_subfolds": "Each original OOF fold is split into five complete-series selection cells.",
    "bootstrap_replicates": "Paired complete-series cluster bootstrap replicates.",
}


__all__ = [
    "CONFIG_HELP",
    "MODEL10_VERSION",
    "Model10Config",
    "make_model10_config",
]
