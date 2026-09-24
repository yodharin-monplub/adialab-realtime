Model57: supervised voter on history-whitened innovation statistics.

Model51 used AR(5) rank-whitened residuals to build an unsupervised Bayes
factor, which failed. Model57 tests the different hypothesis that supervised
trees can learn directed break signatures from a stronger, two-stage
whitening: choose AR order 0/1/2/4/8/12 by BIC on clean history, forecast
one-step residuals, normalize them by a forward-only EWMA variance forecast,
then map them through the empirical CDF of clean-history normalized
innovations. Every transform parameter is fitted on the guaranteed clean
history only. Online residual and variance recursions consume only previous
online points and the current point. A prefix-invariance test runs before
training.

Seven innovation channels—location, squared, absolute, lag-one product,
upper/lower tail rates, and absolute change—are centered/scaled on their
historical behavior. Their standardized suffix sums at 16, 32, 64, 128,
256, 512 and all-observed-prefix ages, plus maximum absolute scale evidence
and five clean-history descriptors, form 61 new causal features.

The 61 columns are appended to Model20's original 256 fold-specific
detector columns. The Model20 400,000 real rows, 400,000 aligned synthetic
rows, 25% synthetic mass share, depth-3/800-round XGBoost parameters and
Model39 ensemble composition are otherwise unchanged. Original feature
caches are verified read-only. Held and adjacent stopping series never
enter fitting rows or synthetic donors. Pilot folds 0-1 select a blend
weight from 0.05, 0.10, 0.20, 0.40; if mean gain is <+0.0001, stop.
Otherwise require confirmation and five-fold held-series OOF TSAUC >=0.640,
confirmation gain >=0.0005, at least 2/3 wins, positive paired series-
bootstrap lower bound, and positive gains in both fixed series-ID halves.

Inherited feature selection and ensemble are development-informed, so this
is exploratory OOF. Sealed outer holdout and competition test labels remain
closed. No cloud training, packaging, upload or submission. CPU only.

Motivation: history-fitted whitening can expose changes in a series'
conditional law, as discussed in the CrunchDAO community thread
https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206.
Our feature design and supervised fit are independent; that discussion
does not establish their performance here.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model57-mpl .venv/bin/python -u models/model57/run_model57.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model57-mpl .venv/bin/python -u models/model57/run_model57.py report
