Model70: larger real-data LightGBM with the established whitened feature bank.

Model62 fits 400,000 sampled real rows plus 400,000 aligned-positive synthetic
rows from three permitted training folds. About 2.5 million real rows are
eligible per fit. Model70 retains the exact original 400,000 and adds 800,000
distinct real rows using Model49's frozen per-fold seed and exclusion rule.
The same 400,000 synthetic rows, 40% synthetic mass within each (time,label)
cell, 317 causal detector-plus-whitened features, 800-tree LightGBM recipe,
held fold and adjacent stopping-fold exclusions, and Model39 postprocessing
remain unchanged. This tests whether more real coverage helps LightGBM even
though Model49's analogous XGBoost test failed.

New-voter fractions 0.50/1.00 and final voter-replacement weights 0.50/1.00
are fixed before fitting. Pilot folds 0-1 need mean gain >=+0.0001 versus
Model62 to train confirmation folds 2-4. Promotion requires confirmation and
full held-series OOF TSAUC >=0.640, confirmation gain >=+0.0005, at least
2/3 wins, positive paired-series bootstrap lower 95% bound, and positive
gains in both fixed ID halves. Inherited feature selection and the ensemble
were development-informed, so scores are exploratory. The sealed outer
holdout and competition test labels remain closed.

CPU only with four LightGBM threads. The runner is fold-checkpointed and has
a 15-hour local timeout. Peak RAM is expected to be several GiB; disk has
at least a 2 GiB reserve. No automatic packaging, upload or cloud submission.

Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model70-mpl .venv/bin/python -u models/model70/run_model70.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model70-mpl .venv/bin/python -u models/model70/run_model70.py report
