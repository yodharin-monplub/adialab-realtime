Model67: fixed three-model held-series ensemble.

Use the already-frozen final held-series OOF prediction from each of Models
62, 63 and 64, with weights 0.50, 0.25 and 0.25 respectively. Each parent
fit excluded the target series' fold and adjacent stopping fold. This model
does no new fitting and touches neither the sealed outer holdout nor test
labels. No weights are optimized on Model67's scoring data. The component
predictions share training data and a Model39-derived stack, so this is a
small variance-reduction test, not a claim of independent models.

Evaluate all five held-series OOF folds and confirmation folds 2-4 against
Model62, with a paired-series bootstrap lower 95% bound and two fixed
series-ID halves. Promotion requires full and confirmation TSAUC >=0.640,
confirmation gain >=0.0005, at least 2/3 wins, positive bootstrap lower
bound, and positive gains in both halves. Prior model selection reused
development folds; results are exploratory and not a fresh holdout.

The fixed weights avoid an optimism trap discussed by other competitors:
https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206

CPU only. No cloud packaging, upload or submission is automatic.
Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model67-mpl timeout 2h .venv/bin/python -u models/model67/run_model67.py
