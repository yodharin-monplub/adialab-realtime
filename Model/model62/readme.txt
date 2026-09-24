Model62: LightGBM with stronger aligned-synthetic training mass.

Model59's original-plus-whitened LightGBM made the largest recent gain, but
the full OOF remains below 0.640 and folds 1/3 are weak. Model21's prior
XGBoost synthetic-weight study found a small ensemble gain at 40% mass over
25%, without robust promotion. Model62 tests that single training-weight
change for LightGBM. It retains Model59's 800-tree/31-leaf standard recipe,
the same 256 selected original plus 61 whitened causal features, 400,000
real and 400,000 aligned synthetic rows, donor exclusions, fitting rows,
random seeds and four CPU threads. Only the synthetic mass share changes
from 25% to 40%; real/synthetic feature caches are reused read-only.

The candidate is either this new voter's probabilities alone or a 50/50
average with Model59's frozen 25%-mass LightGBM. Each replaces 0.25, 0.50,
0.75 or 1.00 of the Model26 voter before the unchanged Model39 stack and
causal smoother. Eight choices are frozen before fitting. Pilot folds 0-1
select one; mean gain must be >=0.0001 to train confirmation folds 2-4.
Promotion requires confirmation and full five-fold TSAUC >=0.640,
confirmation gain >=0.0005, 2/3 wins, positive paired-series bootstrap
lower 95% bound, and positive gains in both fixed ID halves.

These are exploratory development held-series OOF scores. The sealed outer
holdout and competition test labels remain closed. No cloud packaging,
upload or submission is automatic. Expected local runtime is well below
the user's 10-hour limit; the runner checkpoints every fold and stops at
the pilot gate if unsuccessful.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model62-mpl .venv/bin/python -u models/model62/run_model62.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model62-mpl .venv/bin/python -u models/model62/run_model62.py report
