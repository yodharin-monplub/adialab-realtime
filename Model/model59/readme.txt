Model59: regularized LightGBM on original plus whitened causal features.

Model57's history-whitened innovation columns helped when appended to the
strong original detector bank, but the XGBoost gain was uneven. Model58
showed that the whitened bank alone is much weaker, so this study retains
both feature groups and changes only the learner: the regularized LightGBM
recipe previously used in Model44. One model per fold uses Model20's same
400,000 real rows, 400,000 aligned synthetic rows, 25% synthetic mass,
held/stopping exclusions and unchanged feature-cache ordering. Its 317
causal columns are the original 256 fold-selected detector columns plus
Model57's 61 frozen whitened columns. All caches are reused read-only and
checksummed; no raw feature recomputation or new synthetic data is needed.

Model44's 800-tree, 31-leaf, strongly regularized CPU recipe is fixed.
The new probability replaces a fraction 0.05, 0.10, 0.20 or 0.40 of
Model26's voter column before the frozen Model39 stack and smoother.
Pilot folds 0-1 select one weight; if mean gain is below +0.0001, stop.
Otherwise train folds 2-4 and require confirmation and five-fold TSAUC
>=0.640, confirmation gain >=0.0005, at least 2/3 wins, positive paired
series-bootstrap 95% lower bound, and positive gains in two fixed ID halves.

Inherited feature selection, ensemble, and learner choice are development-
informed; this is exploratory held-series OOF. The sealed outer holdout and
competition test labels remain closed. No cloud training, packaging, upload
or submission. CPU-only, four LightGBM threads.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model59-mpl .venv/bin/python -u models/model59/run_model59.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model59-mpl .venv/bin/python -u models/model59/run_model59.py report
