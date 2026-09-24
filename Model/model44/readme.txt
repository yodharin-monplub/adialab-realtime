Model44: regularized LightGBM synthetic-data diversity voter.

The current best backbone uses XGBoost depth diversity.  This experiment
holds the five series folds, 256 selected features, real rows, synthetic
splices, cell weights and causal inference contract fixed, but trains a
regularized leaf-wise LightGBM pair.  Folds 0-1 select one conservative blend
weight from a frozen grid; folds 2-4 are untouched confirmation.

The selected component blend replaces the Model26 XGBoost probability column
inside the frozen ensemble.  The independently developed Model35
anti-residual delta and confirmed Model38 causal smoother are then applied.
The unadjusted Model26-column replacement is retained as a diagnostic.

CPU only, four LightGBM threads.  No packaging, upload or cloud submission.
