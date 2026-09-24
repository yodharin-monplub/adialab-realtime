Model61: extra-randomized LightGBM on original plus whitened causal features.

Model59's standard LightGBM improved full held-series OOF but its weight-1
continuation (Model60) regressed on fold 3. LightGBM's documented extra_trees
mode draws one random threshold per feature at each split, which can reduce
overfitting and produce a complementary learner. This study changes only
extra_trees=True and extra_seed=61 from Model59's 800-tree, 31-leaf recipe.
The same 256 fold-selected original plus 61 frozen whitened features, same
400,000 real and 400,000 aligned synthetic rows, same 25% synthetic mass,
donor exclusions and four CPU threads are retained. All feature caches are
reused read-only and verified; no new synthetic data is generated.

The candidate is either the new voter's probabilities alone or their 50/50
average with Model59's frozen LightGBM probabilities. Each replaces 0.25,
0.50, 0.75 or 1.00 of the Model26 voter before the unchanged Model39 stack
and smoother. Eight choices are prespecified. Pilot folds 0-1 select one;
if its mean gain is below +0.0001, stop. Confirmation folds 2-4 require
confirmation and full five-fold TSAUC >=0.640, gain >=0.0005, 2/3 wins,
positive paired-series bootstrap lower 95% bound and positive gains in both
fixed ID halves.

These are exploratory development held-series OOF scores. The sealed outer
holdout and competition test labels remain closed. No cloud packaging,
upload or submission is automatic.

Source: LightGBM parameters, extra_trees and extra_seed:
https://lightgbm.readthedocs.io/en/latest/Parameters.html

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model61-mpl .venv/bin/python -u models/model61/run_model61.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model61-mpl .venv/bin/python -u models/model61/run_model61.py report
