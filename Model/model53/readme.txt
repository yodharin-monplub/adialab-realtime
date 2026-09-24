Model53: late-horizon Fisher feature reselection for the synthetic XGBoost voter.

An exact metric audit of Model39's held-series OOF found that online steps
160-640 carry about 69% of TSAUC's positive-negative pair weight. The original
XGBoost voter's 256 Fisher-selected features were chosen for the whole online
horizon. Model53 re-ranks the already-causal 439-feature Model20 union using
only fitting-role real rows at steps 160-639, with the original opposite-class
per-time weights, then retains 256 features per fold. The held fold and
adjacent stopping fold are excluded from feature selection, fitting rows,
synthetic donors, and score evaluation.

The 400,000 real rows, original 400,000 aligned synthetic rows, 25% synthetic
mass share, depth-3 XGBoost parameters, 800 rounds, and Model39 composition
are unchanged. Model47 already computed the 439 causal synthetic columns for
these same examples, so this experiment reuses those verified read-only caches.
Only the 256-column selection rule is new. This is exploratory OOF because
the 439-column union and inherited ensemble were development-informed.

Blend weights 0.05, 0.10, 0.20, 0.40 replace Model26's XGBoost probability
column before the frozen stack and causal smoother. Pilot folds 0-1 select
one weight. If mean pilot gain is below +0.0001, stop. Otherwise train folds
2-4 and require confirmation and five-fold TSAUC >=0.640, confirmation gain
>=0.0005, at least 2/3 wins, positive 95% series-bootstrap lower bound and
positive gains in both fixed series-ID halves. Sealed outer holdout and test
labels remain closed. No cloud training, packaging, upload or submission.

CPU-only, four XGBoost threads. Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model53/run_model53.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model53/run_model53.py report
