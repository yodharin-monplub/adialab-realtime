Model52: broader synthetic donor diversity for the proven XGBoost voter.

Model20 used 1,000 distinct break-series recipients per fold although about
1,900 were eligible. Model52 freezes 1,800 distinct recipients from the same
three permitted fitting folds, each paired with a different post-break donor
with reuse <=5. It then samples the same 400,000 synthetic rows from the
larger composite pool. Real fitting rows (400,000), 25% mass-preserving
synthetic weight share, 256 fold-specific causal detector features, original
depth-3 XGBoost recipe, 800 rounds, and the strong Model39 ensemble are kept.
Only donor breadth and the resulting synthetic examples change. No validation
or adjacent stopping-fold series is a donor or fitting row.

Blend weights 0.05, 0.10, 0.20, 0.40 replace the Model26 XGBoost probability
column before the frozen ensemble and causal smoother. Pilot folds 0-1 choose
one weight. If their mean gain is below +0.0001, stop. Otherwise fit folds
2-4 as confirmation. Success requires confirmation and full five-fold mean
TSAUC >=0.640, confirmation gain >=0.0005 over Model39, at least 2/3 wins,
positive 95% series-bootstrap lower bound, and positive gains in both fixed
series-ID halves.

Every trained prediction is held-series OOF, but inherited feature selection
and ensemble weights are development-informed. The sealed outer holdout and
competition test labels remain closed. No cloud upload/submission is automatic.

CPU-only: four feature workers and four XGBoost threads. Feature caches and
fits checkpoint by fold and resume. Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model52/run_model52.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model52/run_model52.py report
