Model90: repaired execution of Model89's Discord Bayesian-evidence protocol.

Model89 froze successfully, built and validated both real and fold-0 synthetic
evidence caches, and reconstructed Model84 exactly. It then stopped before the
first fit because its fold-specific 319-column synthetic matrix was indexed
with global cross-fold union positions. Model90 preserves Model89 unchanged
and changes only that execution mapping: the 317 fold-local parent columns and
the requested scale/dependence evidence columns are projected explicitly into
the union contract before the unchanged fitter selects them.

The scientific experiment is otherwise identical: history-only Model57
whitening; separate inverse-gamma scale and +/-AR(1) dependence Bayes factors;
uniform mass over every candidate tau represented by nodes 1,2,3,4,6,...,768
plus the final partial cell; scale-only, dependence-only and both-channel
LightGBM arms; 400,000 real plus 400,000 aligned synthetic rows; 40% synthetic
TSAUC-cell mass; Model84 baseline; pilot folds 0--1 and untouched confirmation
folds 2--4. The fixed promotion gates and no-cloud rule are unchanged.

The Model89 amplitude audit is diagnostic only. It found fold-stable median
absolute delta-ACF1 0.065 and p90 0.194, strongly supporting the fixed
0.06/0.12/0.20 dependence grid. Adaptive whitening reduced measured persistent
scale amplitudes (median absolute log-sigma ratio 0.025), so scale remains a
separate ablation rather than being forced into the combined arm.

CPU only. Prefix, prior-mass, finite-output, parent-fingerprint, and exact
Model84 baseline checks remain mandatory. No public/test labels, packaging,
upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model90-mpl .venv/bin/python -u models/model90/run_model90.py
