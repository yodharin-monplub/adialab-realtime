"""User-editable configuration for the Model 4 notebook.

Edit constants in this file, restart the notebook kernel, and run from Cell 1.
Existing checkpoints are protected by configuration hashes: incompatible
changes stop with an explanatory error instead of silently mixing runs.
"""

from __future__ import annotations

from pathlib import Path

from models.model4.feature_selection import FeatureSelectionConfig
from models.model4.model4_pipeline import Model4Config


# ---------------------------------------------------------------------------
# Complete-series splitting
# ---------------------------------------------------------------------------
# Split complete series into this many outer groups. One group is reserved as
# the final holdout, so 5 outer folds gives an approximately 20% holdout.
OUTER_FOLDS = 5

# Zero-based outer-fold number used as the sealed holdout. With OUTER_FOLDS=5,
# valid choices are 0, 1, 2, 3, or 4. Keep this fixed: changing it after seeing
# a score would turn the holdout into tuning data.
HOLDOUT_FOLD = 0

# Random seed that assigns complete series to outer folds.
OUTER_SEED = 73_051

# Number of development cross-validation folds. Each model fit uses one fold
# for scoring, the next fold for early stopping, and the remaining folds for
# training. Therefore this must be at least 3.
CV_FOLDS = 5

# Random seed for assigning non-holdout series to development CV folds.
CV_SEED = 84_127


# ---------------------------------------------------------------------------
# Detector-bank selection
# ---------------------------------------------------------------------------
# Include a detector bank only when its mean unseen-fold ordinary ROC AUC is
# strictly greater than this value. ROC AUC is used for detector screening;
# TSAUC remains the competition metric used for combined models.
DETECTOR_ROC_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# LightGBM fitting and uncertainty
# ---------------------------------------------------------------------------
# Deterministic fraction of eligible development rows used to fit LightGBM.
# Validation and scoring still use every row in their assigned folds.
TRAIN_SAMPLE_FRACTION = 0.20

# Seed controlling which development rows enter the training sample.
ROW_SAMPLE_SEED = 99_173

# Maximum trees and early-stopping patience for one-detector screening models.
SCREEN_NUM_BOOST_ROUND = 500
SCREEN_EARLY_STOPPING_ROUNDS = 60

# Maximum trees and early-stopping patience for the all-feature baseline.
COMBINED_NUM_BOOST_ROUND = 1_000
COMBINED_EARLY_STOPPING_ROUNDS = 100

# Number of complete-series bootstrap samples used for the final holdout SE
# and percentile 95% confidence interval. Larger values are more stable but
# take longer after the final predictions have been created.
BOOTSTRAP_REPLICATES = 200


# ---------------------------------------------------------------------------
# Nested feature-selection search
# ---------------------------------------------------------------------------
# Ranking algorithms compared using development-fold TSAUC. ``gain`` and
# ``split`` are LightGBM importances, ``consensus`` combines both rankings, and
# ``gain_corr`` prioritizes high-gain features while removing near-duplicates.
FEATURE_SELECTION_METHODS = (
    "gain",
    "split",
    "consensus",
    "gain_corr",
)

# Number of leading features tested from every ranking. Counts larger than the
# selected detector union are rejected rather than silently reduced.
FEATURE_COUNTS = (32, 64, 128, 256, 512)

# Maximum trees and early-stopping patience for each method/count candidate.
FEATURE_SELECTION_NUM_BOOST_ROUND = 750
FEATURE_SELECTION_EARLY_STOPPING_ROUNDS = 75

# LightGBM seed used by feature-subset candidate models.
FEATURE_SELECTION_MODEL_SEED = 106_031

# Correlation pruning first examines this many top gain-ranked features.
CORRELATION_POOL_SIZE = 1_024

# Number of training-role rows sampled to estimate feature correlations.
CORRELATION_SAMPLE_ROWS = 4_096

# A later feature is treated as redundant when its absolute correlation with a
# previously retained feature reaches this value.
CORRELATION_ABS_THRESHOLD = 0.98

# Seed controlling the correlation-row sample and deterministic ranking work.
FEATURE_SELECTION_RANKING_SEED = 106_013


def make_model_config(project_root: str | Path) -> Model4Config:
    """Build the detector/model protocol from the editable constants above."""

    return Model4Config(
        project_root=Path(project_root).resolve(),
        outer_folds=OUTER_FOLDS,
        holdout_fold=HOLDOUT_FOLD,
        outer_seed=OUTER_SEED,
        development_folds=CV_FOLDS,
        development_seed=CV_SEED,
        train_sample_fraction=TRAIN_SAMPLE_FRACTION,
        row_sample_seed=ROW_SAMPLE_SEED,
        detector_roc_threshold=DETECTOR_ROC_THRESHOLD,
        screen_num_boost_round=SCREEN_NUM_BOOST_ROUND,
        screen_early_stopping_rounds=SCREEN_EARLY_STOPPING_ROUNDS,
        combined_num_boost_round=COMBINED_NUM_BOOST_ROUND,
        combined_early_stopping_rounds=COMBINED_EARLY_STOPPING_ROUNDS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )


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
