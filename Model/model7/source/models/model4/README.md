# Model 4: detector and nested feature selection

`model4.ipynb` is the runnable experiment. It imports the repository's causal
detectors and feature caches; it does not duplicate detector implementations or
use pretrained models.

## Code layout

- `config.py` is the only file normally edited. It controls the outer split,
  development-CV `k`, detector ROC threshold, feature methods/counts, seeds,
  boosting limits, correlation pruning, and bootstrap size.
- `model4_pipeline.py` loads all 43 detector banks, creates series-level
  splits, screens detector banks, and fits the all-feature baseline.
- `feature_selection.py` ranks features, runs the method/count search, freezes
  one winner, fits it, and evaluates the sealed holdout.
- `model4.ipynb` calls those modules one stage per documented cell.

Restart the notebook kernel after editing `config.py`. `CV_FOLDS` must be at
least 3 so a scoring fold, early-stopping fold, and training folds remain
disjoint. The default detector rule is mean unseen-fold ordinary ROC AUC
strictly greater than `DETECTOR_ROC_THRESHOLD = 0.50`.

## Workflow

1. Load only `train/` and validate row-aligned caches for all 28 core-lineage
   and 15 candidate-lineage detector banks.
2. Assign complete series to one of `OUTER_FOLDS` as the holdout (20% by
   default). Independently split the remaining development series into
   `CV_FOLDS` folds.
3. Screen every detector alone. Each fit uses one untouched score fold, the
   next cyclic fold for early stopping, and all remaining folds for training.
4. Freeze the selected detector union and train an all-feature joint LightGBM
   baseline across development folds.
5. From each fold's baseline model, create four rankings:

   - `gain`: LightGBM gain importance;
   - `split`: LightGBM split-count importance;
   - `consensus`: sum of gain and split rank positions;
   - `gain_corr`: gain order with highly correlated leading features pruned.

   Correlation is estimated from that fold's training-role rows only.
6. Test every configured ranking method at 32, 64, 128, 256, and 512 features.
   Each of the 20 default candidates receives `CV_FOLDS` full unseen-fold
   TSAUC scores.
7. Choose the largest mean TSAUC, then lower fold SE, fewer features, and
   configured method order. All fold-specific winning rankings are
   Borda-aggregated into one final feature list.
8. Fit that one frozen subset on development rows with the median winning CV
   round count. Only this winner is scored on holdout. The report includes
   TSAUC and a complete-series bootstrap SE/95% interval.

All candidate summaries are retained because they prove how the winner was
chosen, but only `frozen_winner.json` and `winning_features.npz` define the
final method and features. Losing candidates are never scored on holdout.

## Leakage and interpretation

The outer holdout is never used for detector selection, feature ranking,
method/count selection, early stopping, or round selection. Within the feature
search, the score fold is excluded from ranking-model fitting and early
stopping. Detector banks were selected earlier across the full development
set, so candidate CV scores are selection diagnostics rather than an unbiased
final estimate. The single outer-holdout TSAUC is the approximate submission
score; do not change the winner after reading it.

## Run and resume

Open `models/model4/model4.ipynb` and run top to bottom. Defaults require up to
321 LightGBM fits from scratch: 215 detector fits, 5 all-feature baseline fits,
100 feature-subset fits, and 1 final fit. Existing compatible checkpoints are
reused automatically.

Outputs are under `models/model4/all_detector_artifacts/`. Feature-search files
are isolated below `feature_selection/`, including per-fold rankings,
candidate checkpoints, `search_results.json`, the frozen winner, final model,
holdout predictions, and `HOLDOUT_REPORT.md`. Configuration hashes prevent
mixing incompatible runs. Preserve or rename the old artifact directory before
starting a changed protocol.

## Complete source copy

[`source/`](source/) contains a hash-manifested mini-repository with this model,
its shared root pipelines, all imported detectors, and regression tests. It
excludes generated caches, fitted models, predictions, logs, and training data.
