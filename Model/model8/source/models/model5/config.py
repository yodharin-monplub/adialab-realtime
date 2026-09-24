"""User-editable configuration for the Model 5 notebook.

Edit constants here, restart the notebook kernel, and run from Cell 1. Changed
model protocols are routed to separate run directories. Within one run, saved
stage contracts reject incompatible checkpoints instead of silently mixing
different feature searches.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from models.model5.feature_selection import FeatureSelectionConfig
from models.model5.model5_pipeline import Model5Config


# Optional name for a separate run below ``models/model5/runs/``. Leave this
# as None to reuse the legacy ``artifacts/`` directory when it is compatible,
# or automatically choose a deterministic directory when settings differ.
# Example manual value: ``RUN_NAME = "tsauc_055_experiment"``.
RUN_NAME: str | None = None


# ---------------------------------------------------------------------------
# Complete-series splitting
# ---------------------------------------------------------------------------
# Model 5 uses one outer split directly: one fold is sealed and every other
# fold becomes one development-CV fold. Therefore OUTER_FOLDS must always be
# CV_FOLDS + 1. The defaults give one holdout plus five CV folds.
OUTER_FOLDS = 6
HOLDOUT_FOLD = 0
OUTER_SEED = 20_260_826
CV_FOLDS = 5

# Retained in the serialized protocol for compatibility. Because Model 5 maps
# non-holdout outer folds directly to CV folds, OUTER_SEED controls assignment.
CV_SEED = 20_260_839


# ---------------------------------------------------------------------------
# TSAUC detector-bank selection
# ---------------------------------------------------------------------------
# Model 5 never uses ordinary row ROC to select detectors. A bank is included
# only when:
#
#   mean unseen-fold TSAUC - DETECTOR_SE_MULTIPLIER * TSAUC fold SE
#       > DETECTOR_TSAUC_THRESHOLD
#
# For example, a multiplier of 0.5 applies the requested half-SE penalty. The
# threshold below can be changed independently; a changed model protocol is
# automatically routed to a separate run directory.
DETECTOR_TSAUC_THRESHOLD = 0.55
DETECTOR_SE_MULTIPLIER = 0.50


# ---------------------------------------------------------------------------
# LightGBM fitting and final uncertainty
# ---------------------------------------------------------------------------
# Only this deterministic fraction of eligible rows fits LightGBM. Every row
# in the separate early-stop and score folds is still evaluated.
TRAIN_SAMPLE_FRACTION = 0.20
ROW_SAMPLE_SEED = 99_173

# Maximum trees and early-stopping patience for one-detector models.
SCREEN_NUM_BOOST_ROUND = 500
SCREEN_EARLY_STOPPING_ROUNDS = 60

# Maximum trees and early-stopping patience for the all-feature baseline.
COMBINED_NUM_BOOST_ROUND = 1_000
COMBINED_EARLY_STOPPING_ROUNDS = 100

# Complete-series bootstrap samples for the final holdout TSAUC SE and 95% CI.
BOOTSTRAP_REPLICATES = 200

# Independent per-series feature extraction releases the GIL in NumPy/SciPy.
# A same-series benchmark measured two workers as faster than 1, 4, 8, or 12
# on this machine because several kernels also use internal threads. This
# affects runtime only; cache signatures still depend solely on detector code,
# names, and bank order.
CACHE_BUILD_WORKERS = 2


# ---------------------------------------------------------------------------
# Nested feature-selection search
# ---------------------------------------------------------------------------
# ``gain`` and ``split`` use LightGBM importance, ``consensus`` combines their
# rank positions, and ``gain_corr`` removes near-duplicates from gain order.
FEATURE_SELECTION_METHODS = (
    "gain",
    "split",
    "consensus",
    "gain_corr",
)

# Test each leading-feature count for every ranking method. Oversized counts
# raise an error instead of being silently reduced.
FEATURE_COUNTS = (32, 64, 128, 256, 512)

# Candidate-model tree limit, early-stopping patience, and deterministic seed.
FEATURE_SELECTION_NUM_BOOST_ROUND = 750
FEATURE_SELECTION_EARLY_STOPPING_ROUNDS = 75
FEATURE_SELECTION_MODEL_SEED = 116_031

# Correlation pruning examines leading gain features on sampled training-role
# rows. Features at or above the absolute threshold are treated as redundant.
CORRELATION_POOL_SIZE = 1_024
CORRELATION_SAMPLE_ROWS = 4_096
CORRELATION_ABS_THRESHOLD = 0.98
FEATURE_SELECTION_RANKING_SEED = 116_013


def _configured_model(project_root: Path, artifact_directory: str) -> Model5Config:
    """Build one config before its automatic run directory is resolved."""

    return Model5Config(
        project_root=project_root,
        artifact_directory=artifact_directory,
        outer_folds=OUTER_FOLDS,
        holdout_fold=HOLDOUT_FOLD,
        outer_seed=OUTER_SEED,
        development_folds=CV_FOLDS,
        development_seed=CV_SEED,
        train_sample_fraction=TRAIN_SAMPLE_FRACTION,
        row_sample_seed=ROW_SAMPLE_SEED,
        detector_roc_threshold=DETECTOR_TSAUC_THRESHOLD,
        detector_standard_error_multiplier=DETECTOR_SE_MULTIPLIER,
        screen_num_boost_round=SCREEN_NUM_BOOST_ROUND,
        screen_early_stopping_rounds=SCREEN_EARLY_STOPPING_ROUNDS,
        combined_num_boost_round=COMBINED_NUM_BOOST_ROUND,
        combined_early_stopping_rounds=COMBINED_EARLY_STOPPING_ROUNDS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        cache_build_workers=CACHE_BUILD_WORKERS,
    )


def _run_directory(project_root: Path, provisional: Model5Config) -> str:
    """Reuse a compatible legacy run or derive a safe path for changed settings."""

    if RUN_NAME is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", RUN_NAME):
            raise ValueError(
                "RUN_NAME must contain only letters, numbers, dot, dash, or underscore"
            )
        return f"models/model5/runs/{RUN_NAME}"

    legacy_directory = "models/model5/artifacts"
    legacy_manifest = project_root / legacy_directory / "experiment_manifest.json"
    if not legacy_manifest.exists():
        return legacy_directory
    saved = json.loads(legacy_manifest.read_text(encoding="utf-8"))
    if saved.get("protocol") == provisional.protocol():
        return legacy_directory

    # The artifact path itself must not influence the run identity.
    identity = dict(provisional.protocol())
    identity.pop("artifact_directory")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    threshold = f"{DETECTOR_TSAUC_THRESHOLD:g}".replace("-", "m").replace(".", "p")
    penalty = f"{DETECTOR_SE_MULTIPLIER:g}".replace("-", "m").replace(".", "p")
    run_label = f"cv{CV_FOLDS}_tsauc{threshold}_se{penalty}_{digest}"
    return f"models/model5/runs/{run_label}"


def make_model5_config(project_root: str | Path) -> Model5Config:
    """Build Model 5 and route incompatible settings to a separate run."""

    resolved_root = Path(project_root).resolve()
    legacy = _configured_model(resolved_root, "models/model5/artifacts")
    artifact_directory = _run_directory(resolved_root, legacy)
    return replace(legacy, artifact_directory=artifact_directory)


FEATURE_SELECTION_CONFIG = FeatureSelectionConfig(
    methods=FEATURE_SELECTION_METHODS,
    feature_counts=FEATURE_COUNTS,
    num_boost_round=FEATURE_SELECTION_NUM_BOOST_ROUND,
    early_stopping_rounds=FEATURE_SELECTION_EARLY_STOPPING_ROUNDS,
    model_seed=FEATURE_SELECTION_MODEL_SEED,
    correlation_pool_size=CORRELATION_POOL_SIZE,
    correlation_sample_rows=CORRELATION_SAMPLE_ROWS,
    correlation_abs_threshold=CORRELATION_ABS_THRESHOLD,
    ranking_seed=FEATURE_SELECTION_RANKING_SEED,
)
