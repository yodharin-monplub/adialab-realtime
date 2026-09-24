# Model 5: TSAUC detector and nested feature selection

Model 5 now follows Model 4's detector → feature search → single holdout-winner
workflow. Its defining difference is detector selection with time-stratified
ROC AUC (`TSAUC`), including the requested uncertainty penalty.

## Code layout

- `config.py` contains every normally edited split, threshold, training, and
  feature-search setting, with code-level explanations.
- `model5_pipeline.py` configures the shared all-43-detector pipeline for
  Model 5's direct outer-fold CV and TSAUC detector metric.
- `feature_selection.py` exposes the exact shared leakage-safe ranking and
  winner-freezing implementation used by Model 4.
- `model5.ipynb` is the documented, resumable runner and includes a Mermaid
  training/split flowchart.

No detector implementation is copied and no pretrained model is used.

## Detector rule

Each bank receives one full unseen TSAUC score per development fold. The
detector rule is:

```text
mean unseen-fold TSAUC - configured SE multiplier * fold TSAUC SE
    > configured TSAUC threshold
```

`DETECTOR_SE_MULTIPLIER` and `DETECTOR_TSAUC_THRESHOLD` control the two numeric
terms; a multiplier of `0.5` gives the requested half-SE penalty. The
comparison is strict, so equality is excluded. Ordinary row ROC is recorded
only as a diagnostic and never decides Model 5 detector inclusion.

Within each detector fit, the scoring fold is untouched, the next cyclic fold
is used for early stopping, and all other development folds train LightGBM.

## Split and feature search

The defaults assign complete series once into six folds:

- outer fold 0 is the sealed holdout;
- outer folds 1–5 directly become development folds 0–4.

`OUTER_FOLDS` must equal `CV_FOLDS + 1`. Rows from the same series never cross
roles.

After detector banks are frozen, a joint all-selected-feature baseline creates
fold-specific importance rankings. Development CV tests four methods—gain,
split count, gain/split consensus, and correlation-pruned gain—at 32, 64, 128,
256, and 512 features. The winner has the largest mean TSAUC, then lower TSAUC
SE, fewer features, and earlier configured method order.

Only that winner is Borda-aggregated, frozen, fitted on development data with a
fixed CV-derived tree count, and evaluated on holdout. The final report includes
holdout TSAUC and a complete-series bootstrap SE and 95% interval. Losing
candidates are not scored on holdout.

## Run and checkpoints

Restart the kernel after editing `config.py`, open `model5.ipynb`, and run top
to bottom. Defaults require up to 321 LightGBM fits from scratch: 215 detector
fits, 5 all-feature baseline fits, 100 feature-subset fits, and 1 final fit.

Existing detector, split, and family-model artifacts under `artifacts/` are
preserved. When only the TSAUC cutoff and run path differ, Model 5 hard-links
the compatible train-derived fold models and predictions, then recomputes
mutable detector summaries against the new cutoff. This avoids 215 redundant
fits without modifying the original run. Old core/candidate architecture
outputs are no longer used for winner selection. New feature-search artifacts
are isolated below the active run's `feature_selection/` directory.

Configuration hashes prevent incompatible checkpoint mixing. `config.py`
reuses the legacy `artifacts/` directory only when its model protocol matches;
otherwise it automatically chooses a deterministic directory below `runs/`.
Set `RUN_NAME` to give a run a manual name. This preserves old runs when CV,
threshold, split, sampling, or training settings change. If only feature-search
settings change inside the same model run, rename that run's
`feature_selection/` directory before rerunning the search.

The outer holdout is never used for detector selection, ranking, method/count
selection, early stopping, or tree-count selection. Do not change the frozen
winner after reading its holdout score.

## Complete source copy

[`source/`](source/) contains a hash-manifested mini-repository with this model,
its shared root pipelines, upstream model code, all imported detectors, and
tests. Generated artifacts and training data are excluded.
