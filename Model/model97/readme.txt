Model97: maximum-over-tau empirical-law cell GLR.

Model96 showed that maximizing scale evidence over candidate onset generalizes
better than uniform Bayes integration. The Discord brief also reports strong
oracle signal from empirical-law cells, especially at short post-break ages.
Model97 divides Model57's history-whitened innovations into eight equal-normal-
probability cells. For every suffix age 4,6,8,12,...,768 and the full observed
prefix, it computes the multinomial generalized likelihood ratio against the
uniform null law and retains the maximum.

This one statistic detects changes in scale, tails, skew and other marginal
shape without a broad feature bundle. It is causal and prefix invariant;
lengths below four are suppressed to limit sparse one-point maxima. Its
label-free pilot standard deviation is 3.123, so native additive weights
0.0005, 0.001, 0.002, 0.004, 0.008, 0.016 and 0.032 are frozen before scoring.
Pilot folds 0--1 require a +0.0001 gain before separate confirmation.

Motivation:
  ../DISCORD_NEXT_MODEL_BRIEF.md
  https://proceedings.mlr.press/v206/puchkin23a.html

CPU only. No public/test labels, cloud activity, packaging, upload, or
submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model97-mpl .venv/bin/python -u models/model97/run_model97.py
