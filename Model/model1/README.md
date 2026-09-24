# Model 1: Pretrained Core/Candidate Baseline

Model 1 is the original equal-logit LightGBM ensemble with separate core and
candidate detector families. The reusable cache, cross-validation, and blend
implementation remains in the repository-root `train_cv.py`, `ensemble_cv.py`,
`feature_pipeline.py`, and `solution.py` modules.

The frozen deployment remains in
[`submissions/submission_1/`](../../submissions/submission_1/). A complete
source-only copy is also stored in [`source/`](source/), including the exact
manifested deployment files. Retained development/bundle drivers are separated
under `source/research/`; no Git history remains to prove those current driver
revisions are the original ones. Trained weights, caches, predictions, and
training data are not source code and are deliberately excluded.

Model 1's exact deployed two-model behavior also depends on its frozen
pretrained `submissions/submission_1/resources/` weights. Those artifacts remain
in the submission bundle and are not duplicated into this source-only folder;
the weight-free fallback trains a different, core-only model.
