Model105: sparse-tail repair for nested empirical-null scale-GLR calibration.

Model104 correctly refused to score because exact-time null tables become empty
at the longest online steps: different held folds have different maximum series
lengths. Model105 preserves the same nested outer/inner fold exclusions and
weight grid, but pools the nearest observable online steps symmetrically until
at least 500 target-zero reference rows are available. At well-supported times
the exact-time table remains unchanged; only sparse late times borrow adjacent
steps. The held fold remains absent from both calibration and weight selection.

This nearest-time pooling is deployable and follows the finite-time calibration
motivation while making the long-horizon boundary explicit. All other Model104
science, promotion gates, and no-cloud policy remain unchanged.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model105-mpl .venv/bin/python -u models/model105/run_model104.py
