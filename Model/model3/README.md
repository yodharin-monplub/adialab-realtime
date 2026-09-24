# Model 3: Feature-Selected Ensemble

Model 3 added leakage-controlled feature selection to the core/candidate
LightGBM ensemble. The experiment driver remains in the repository-root
`feature_selection_experiment.py`; the frozen column contracts and complete
deployment code are retained in
[`submissions/submission_3/`](../../submissions/submission_3/).

A complete source-only copy is stored in [`source/`](source/). It includes the
exact historical submission code, detector snapshot, feature-selection driver,
and the two small frozen column-contract NPZ files. The driver is retained from
the current repository because no Git history survives to prove its exact
historical revision. Generated predictions, feature matrices, trained models,
caches, and training data are excluded.
