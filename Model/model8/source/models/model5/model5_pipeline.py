"""Model 5 configuration over the audited all-detector training pipeline.

Model 5 deliberately reuses Model 4's cache, fitting, checkpoint, and sealed
holdout implementation. Its two default protocol changes are isolated here:

1. one six-way complete-series split supplies one sealed holdout fold and the
   five remaining development folds directly;
2. a detector is selected only when its development-fold mean TSAUC minus half
   of its fold standard error is strictly greater than chance (0.5).

No detector implementation, pretrained model, or holdout outcome is copied
into this module.
"""

from __future__ import annotations

import math
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from models.model4 import model4_pipeline as _shared


MODEL5_VERSION = "model5-tsauc-half-se-sixfold-v1"
MODEL5_FEATURE_SELECTION_VERSION = "model5-nested-feature-selection-v1"


@dataclass(frozen=True)
class Model5Config(_shared.Model4Config):
    """Frozen, serializable protocol for the Model 5 notebook."""

    experiment_version: str = MODEL5_VERSION
    experiment_title: str = "Model 5"
    artifact_directory: str = "models/model5/artifacts"
    outer_folds: int = 6
    holdout_fold: int = 0
    # Fresh split identity for the 10,000-series training refresh (2026-08-26).
    outer_seed: int = 20_260_826
    development_folds: int = 5
    development_seed: int = 20_260_839
    detector_selection_metric: str = "tsauc"
    detector_standard_error_multiplier: float = 0.5
    development_from_outer_folds: bool = True
    screen_model_seed: int = 95_051
    combined_model_seed: int = 95_063
    bootstrap_seed: int = 20_260_823

    @property
    def feature_selection_workflow_version(self) -> str:
        """Label Model 5 feature artifacts without changing its model protocol."""

        return MODEL5_FEATURE_SELECTION_VERSION

    def validate(self) -> None:
        super().validate()
        if self.experiment_version != MODEL5_VERSION:
            raise ValueError("Model5 experiment_version is immutable")
        if self.detector_selection_metric != "tsauc":
            raise ValueError("Model5 detector selection must use TSAUC")
        if (
            not math.isfinite(self.detector_standard_error_multiplier)
            or self.detector_standard_error_multiplier < 0.0
        ):
            raise ValueError("Model5 TSAUC SE multiplier must be non-negative")
        if not self.development_from_outer_folds:
            raise ValueError("Model5 must reuse the non-holdout outer folds")
        if self.outer_folds != self.development_folds + 1:
            raise ValueError(
                "Model5 requires one outer holdout plus CV_FOLDS development folds"
            )


def load_model5_data(config: Model5Config):
    """Load `train/` and the existing causal caches for all 43 detectors."""

    config.validate()
    data = _shared.load_model4_data(config)
    reuse_compatible_detector_checkpoints(config, data)
    return data


def _screen_compatible_protocol(protocol: dict) -> dict:
    """Drop choices that cannot change fitted detector-fold predictions."""

    comparable = dict(protocol)
    # The detector cutoff is applied after every fold prediction is scored.
    # Its value and the destination path cannot change those predictions.
    comparable.pop("artifact_directory", None)
    comparable.pop("detector_roc_threshold", None)
    return comparable


def reuse_compatible_detector_checkpoints(
    config: Model5Config, data: _shared.ExperimentData
) -> dict | None:
    """Hard-link complete train-derived detector folds into a changed-cutoff run.

    This reuse is deliberately conservative: all protocol values except the
    post-fit detector threshold and artifact path must match. Mutable summaries
    and OOF arrays are not linked, so the new threshold is recomputed locally.
    """

    target = config.artifact_path.resolve()
    model5_root = config.project_root / "models/model5"
    candidate_roots = [model5_root / "artifacts"]
    runs_root = model5_root / "runs"
    if runs_root.exists():
        candidate_roots.extend(sorted(path for path in runs_root.iterdir() if path.is_dir()))
    desired_protocol = _screen_compatible_protocol(config.protocol())
    required_banks = {
        "core": tuple(data.core_banks),
        "candidate": tuple(data.candidate_banks),
    }

    source: Path | None = None
    for candidate in candidate_roots:
        if candidate.resolve() == target:
            continue
        manifest_path = candidate / "experiment_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _screen_compatible_protocol(manifest.get("protocol", {})) != desired_protocol:
            continue
        if tuple(manifest.get("core_banks", ())) != required_banks["core"]:
            continue
        if tuple(manifest.get("candidate_banks", ())) != required_banks["candidate"]:
            continue
        complete = True
        for family, banks in required_banks.items():
            for bank in banks:
                directory = candidate / "screen" / family / bank
                required = [directory / "manifest.json"]
                for fold in range(config.development_folds):
                    required.extend(
                        (
                            directory / f"fold_{fold}.json",
                            directory / f"fold_{fold}.prediction.f32.npy",
                            directory / f"fold_{fold}.txt",
                        )
                    )
                if not all(path.exists() for path in required):
                    complete = False
                    break
            if not complete:
                break
        if complete:
            source = candidate
            break

    if source is None:
        return None

    immutable_files_per_bank = 1 + 3 * config.development_folds
    linked_banks = 0
    available_banks = 0
    for family, banks in required_banks.items():
        target_family = target / "screen" / family
        target_family.mkdir(parents=True, exist_ok=True)
        for bank in banks:
            source_bank = source / "screen" / family / bank
            target_bank = target_family / bank
            immutable_names = ["manifest.json"]
            for fold in range(config.development_folds):
                immutable_names.extend(
                    (
                        f"fold_{fold}.json",
                        f"fold_{fold}.prediction.f32.npy",
                        f"fold_{fold}.txt",
                    )
                )
            if target_bank.exists():
                if all((target_bank / name).exists() for name in immutable_names):
                    available_banks += 1
                continue
            temporary = target_family / f".{bank}.reuse_tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            try:
                for name in immutable_names:
                    os.link(source_bank / name, temporary / name)
                temporary.rename(target_bank)
            except OSError:
                shutil.rmtree(temporary, ignore_errors=True)
                print(
                    "Detector checkpoint hard-link reuse was unavailable; "
                    "missing folds will train normally.",
                    flush=True,
                )
                return None
            linked_banks += 1
            available_banks += 1

    report = {
        "source": str(source.relative_to(config.project_root)),
        "target": config.artifact_directory,
        "reason": (
            "all detector-fit settings and splits match; only the post-fit "
            "TSAUC cutoff and artifact path may differ"
        ),
        "banks_available": available_banks,
        "banks_linked_now": linked_banks,
        "immutable_files_per_bank": immutable_files_per_bank,
        "mutable_summaries_and_oof_linked": False,
    }
    report_path = target / "reused_detector_checkpoints.json"
    if report_path.exists():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        # The first call may link banks; later calls merely find them present.
        report["banks_linked_now"] = prior.get("banks_linked_now", linked_banks)
    _shared._json_dump(report_path, report)
    print(
        f"Reusable detector folds available for {available_banks} banks from "
        f"{report['source']}",
        flush=True,
    )
    return report


# Re-export the audited stages under the Model 5 import path. Each function
# receives Model5Config, so manifests, selection rules, paths, and split policy
# are recorded as Model 5 rather than Model 4.
ExperimentData = _shared.ExperimentData
SplitAssignment = _shared.SplitAssignment
architecture_table = _shared.architecture_table
cross_validate_family_models = _shared.cross_validate_family_models
cross_validate_joint_model = _shared.cross_validate_joint_model
detector_is_selected = _shared.detector_is_selected
detector_selection_bound = _shared.detector_selection_bound
evaluate_holdout = _shared.evaluate_holdout
fit_final_models = _shared.fit_final_models
freeze_architecture = _shared.freeze_architecture
prepare_split = _shared.prepare_split
save_selected_detectors = _shared.save_selected_detectors
screen_family = _shared.screen_family
screen_table = _shared.screen_table
split_table = _shared.split_table


__all__ = [
    "MODEL5_VERSION",
    "MODEL5_FEATURE_SELECTION_VERSION",
    "Model5Config",
    "ExperimentData",
    "SplitAssignment",
    "architecture_table",
    "cross_validate_family_models",
    "cross_validate_joint_model",
    "detector_is_selected",
    "detector_selection_bound",
    "evaluate_holdout",
    "fit_final_models",
    "freeze_architecture",
    "load_model5_data",
    "prepare_split",
    "reuse_compatible_detector_checkpoints",
    "save_selected_detectors",
    "screen_family",
    "screen_table",
    "split_table",
]
