Model68: CPU CatBoost diversity voter on original-plus-whitened features.

Model46's CatBoost on 256 selected original detector features contributed
almost nothing. Model59/62 showed that adding 61 causal history-whitened
innovation features materially helped LightGBM. Model68 tests whether the
same 317 features and Model62's 40% aligned-synthetic training mass help
CatBoost enough to supply a complementary voter. It retains Model46's
CatBoost 500-tree/depth-5 recipe and Model20/62's exact 400,000 real plus
400,000 synthetic rows, donor exclusions and fold seeds. Existing feature
caches are reused read-only. No held series is a donor or fitting row.

The new CatBoost probability replaces 5%, 10%, 20% or 40% of Model62's
frozen LightGBM voter probability before the unchanged Model39 stack and
causal smoother. Pilot folds 0-1 select one. If no mean gain over Model62
reaches +0.0001, stop before confirmation folds 2-4. Promotion requires
confirmation and full five-fold TSAUC >=0.640, confirmation gain >=0.0005
over Model62, 2/3 wins, positive paired-series bootstrap lower 95% bound,
and positive gains in both fixed series-ID halves. The model is exploratory
because inherited features and ensemble were developed on these folds.
The sealed outer holdout and competition test labels remain closed.

CPU only, four CatBoost threads. No cloud packaging, upload or submission.
Run/resume from repository root within the 15-hour cap:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model68-mpl timeout 14h .venv/bin/python -u models/model68/run_model68.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model68-mpl .venv/bin/python -u models/model68/run_model68.py report
