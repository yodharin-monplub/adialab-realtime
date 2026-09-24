Model82: cross-fitted pairwise simplex stack over diverse OOF sources.

Recent experiments often improve one held fold and regress another. Model82
learns a single global combination for each held fold using only the other
four folds. Its ten frozen sources are Model67, Model39, Model57, Model59,
Models62-64, Model45's analytic kernel discrepancy, Model48's original
pairwise ranker, and Model81's whitened pairwise ranker. Every source is first
converted to a percentile rank within (fold, online time), preserving its
individual TSAUC and placing all sources on a common scale.

For each development fold, 50,000 positive/negative pairs are sampled exactly
in proportion to that fold's n_positive(t)*n_negative(t) TSAUC mass and
uniformly within time. To predict held fold h, pairwise logistic loss is fit on
the 200,000 pairs from folds other than h. Weights are nonnegative and sum to
one. A fixed L2 penalty 0.25 anchors them to 100% Model67, preventing a noisy
source pool from causing large shifts. No held-fold label influences its
weights, source ranks, or optimization.

All five cross-fitted predictions are evaluated. Promotion requires mean OOF
TSAUC >=0.640, gain >=0.0005 over Model67, at least 3/5 wins, a positive
paired-series bootstrap lower 95% bound, and positive gains in two fixed ID
halves. The source pool and penalty are development-informed, so this is not a
virgin outer holdout; nevertheless every reported row is unseen by its own
meta-fit. The sealed outer holdout and competition labels remain closed.
CPU only; no cloud packaging, upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model82-mpl timeout 3h .venv/bin/python -u models/model82/run_model82.py

