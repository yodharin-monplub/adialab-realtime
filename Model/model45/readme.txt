Model45: causal multiscale kernel discrepancy from the historical reference.

This CPU-only experiment adds a label-free change signal to Model39's held-series
OOF margins. For each series, robustly standardize historical levels and first
differences, embed them with fixed cosine/sine frequencies, and compare the
historical mean embedding to trailing online windows of 32 and 128 points.
Each score uses only the current online prefix. The score is an analytic,
fixed-frequency approximation to a kernel two-sample discrepancy, motivated by
online contrastive and maximum-mean-discrepancy change detection:
https://proceedings.mlr.press/v206/puchkin23a.html
https://arxiv.org/abs/2205.12706

Pilot folds 0-1 set a robust signal scale and select one nonnegative additive
weight from the frozen grid. Held series in folds 2-4 confirm the choice. The
predeclared target is >=0.640 both on the confirmation mean and all five OOF
folds, plus positive whole-series uncertainty and split-half checks.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model45/run_model45.py run

No GPU, new fitted model, packaging, upload, or cloud submission. Development
OOF has been used repeatedly for earlier research; the sealed outer holdout
remains closed. Synthetic training rows are not used in this analytic signal.
