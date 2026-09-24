Model64: LightGBM on the full original detector union plus whitening.

Models59-63 gave LightGBM the 256 original detector columns selected by an
earlier XGBoost model, plus 61 causal history-whitened columns. The real
feature cache already contains 439 original detector columns. XGBoost's
selection need not be optimal for LightGBM, so Model64 tests all 439 plus
the same 61 whitened columns (500 total), while holding Model62's standard
LightGBM recipe and 40% aligned-synthetic mass fixed.

For each fold, the same 400,000 real rows, 400,000 aligned synthetic rows,
donor exclusions, weights and seeds are retained. The original 256
synthetic columns and 61 whitened columns are reused from verified caches;
the 183 missing original columns alone are computed from the frozen causal
feature pipeline and checkpointed by synthetic task. The real 439+61 cache
is reused read-only. Four CPU feature workers and four LightGBM threads.
An initial synthetic task took 0.425 seconds to compute all 183 missing
columns, giving a rough full-run estimate under one hour; a 9-hour process
cap enforces the user's 10-hour runtime limit.

The candidate is either the new 500-column voter's probabilities alone or
a 50/50 average with Model62's frozen 317-column voter. Each replaces
0.50 or 1.00 of Model26's voter before the unchanged Model39 ensemble and
causal smoother. Four choices are frozen before fitting. Pilot folds 0-1
select one; a mean gain below +0.0001 stops before confirmation folds 2-4.
Promotion requires confirmation and full five-fold TSAUC >=0.640,
confirmation gain >=0.0005, 2/3 wins, positive paired-series bootstrap
lower 95% bound, and positive gains in both fixed ID halves.

These are exploratory development held-series OOF scores. The sealed outer
holdout and competition test labels remain closed. No cloud packaging,
upload or submission is automatic.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model64-mpl timeout 9h .venv/bin/python -u models/model64/run_model64.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model64-mpl .venv/bin/python -u models/model64/run_model64.py report
