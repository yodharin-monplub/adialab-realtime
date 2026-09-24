Model46: CPU CatBoost diversity voter on the frozen real-plus-synthetic pool.

Fit five CatBoost models, one per held-series fold, with the same 400,000
sampled real rows, 400,000 aligned-splice rows, selected 256 features, per-cell
synthetic weight share, and causal inference feature contract as Model20.
Only the learner family changes. A conservative blend replaces Model26's
XGBoost probability column inside the frozen ensemble; Model35's existing
anti-residual correction and Model38's confirmed smoother are then applied.

Weights 0.05, 0.10, 0.20, 0.30 are selected on pilot folds 0-1. Folds 2-4
confirm. The predeclared goal is both full five-fold and three confirmation
fold means >=0.640, gain >=0.0005, 2/3 wins, a positive series-bootstrap
lower bound, and positive gains in both fixed ID halves. Prior feature and
ensemble development reused these folds, so this is not a fresh outer test.

CatBoost 1.2.10 is required locally; it has an official CPython 3.14 Linux
wheel (https://pypi.org/project/catboost/1.2.10/). CPU only, four threads.
Run or resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model46/run_model46.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model46/run_model46.py report

No packaging, upload, or cloud submission. The sealed outer holdout remains
closed, and no competition test labels are used.
