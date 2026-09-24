Model65: test a stronger real-plus-synthetic LightGBM training mixture.

Model62's 317-feature standard LightGBM with 40% aligned-synthetic mass has
the best full held-series OOF TSAUC so far (0.637534). Model65 changes only
the mass assigned to the same 400,000 cached synthetic rows: 60% instead of
40%, while retaining the same 400,000 real rows, donor exclusions, 256
selected original plus 61 causally whitened features, model parameters and
fold seeds. Synthetic and real feature caches are reused read-only; there
is no additional data generation or held-series leakage. Four CPU threads.

The new voter alone or its 50/50 average with Model62's frozen voter replaces
0.75 or 1.00 of Model26's voter inside the unchanged Model39 stack. Four
choices are frozen before fitting. Pilot folds 0-1 select one by mean gain
over Model62's already-frozen candidate and stop if gain is <+0.0001.
Confirmation folds 2-4 are untouched by this choice. Promotion requires
confirmation and full five-fold TSAUC >=0.640, confirmation gain >=0.0005
over Model62, at least 2/3 wins, a positive paired-series bootstrap lower
95% bound, and positive gains in both fixed series-ID halves. These are
exploratory held-series OOF scores because earlier work informed the feature
bank and incumbent. The sealed outer holdout and test labels remain closed.
No cloud packaging, upload or submission is automatic.

Research context: other participants report diminishing but nonzero returns
from synthetic augmentation, motivating a direct mass test rather than
another broad feature bank:
https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206
LightGBM's documented sample_weight interface makes the mass change explicit:
https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html

Run/resume from the repository root (bounded well inside 15 hours):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model65-mpl timeout 14h .venv/bin/python -u models/model65/run_model65.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model65-mpl .venv/bin/python -u models/model65/run_model65.py report
