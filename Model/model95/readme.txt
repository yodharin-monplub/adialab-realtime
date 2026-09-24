Model95: maximum-over-tau scale GLR on whitened innovations.

Model94 showed that uniformly integrated scale Bayes evidence has a small but
non-robust effect. Model95 replaces integration with a generalized likelihood
ratio: for each admissible suffix it maximizes the Gaussian variance likelihood
over the unknown post-change variance, then takes the maximum over suffix ages
4,6,8,12,...,768 and the complete observed online prefix. This is two-sided
for variance increases and decreases. Lengths below four are suppressed to
avoid a one-point outlier dominating the maximum.

The feature uses Model57's history-only AR/BIC, forward EWMA, and empirical-
normal innovations. It is causal and prefix-invariant. Its label-free pilot
standard deviation is 1.817 versus 0.571 for Model84, so native-score additive
weights 0.00125, 0.0025, 0.005, 0.01, 0.02 and 0.04 are frozen before scoring.
Pilot folds 0--1 select one weight and require a +0.0001 mean gain before any
separate confirmation model.

Motivation:
  https://arxiv.org/abs/1309.5264
  https://proceedings.mlr.press/v206/puchkin23a.html

CPU only. No public/test labels, cloud activity, packaging, upload, or
submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model95-mpl .venv/bin/python -u models/model95/run_model95.py
