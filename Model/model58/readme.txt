Model58: standalone supervised whitened-innovation voter.

Model57's 61 history-whitened innovation features improved the strong
Model20 voter on some folds, but appending them to 256 original detector
columns gave uneven gains. Model58 isolates the new representation: the
same fold-local 400,000 real rows and 400,000 aligned synthetic rows train
one depth-3, 800-round XGBoost per fold using only those 61 whitened
features. The synthetic mass share stays at 25%. No new features or data
are generated; every real and synthetic feature cache from Model57 is
reused read-only with protocol and content checksums.

The standalone model is intended as a diversity voter, not an assumed
replacement for Model39. Fixed weights 0.05, 0.10, 0.20, 0.40 replace that
fraction of Model26's XGBoost probability column before the unchanged
Model39 stack and causal smoother. Pilot folds 0-1 select one weight; if
mean pilot gain is below +0.0001, stop. Otherwise train folds 2-4 and
require confirmation and five-fold held-series OOF TSAUC >=0.640,
confirmation gain >=0.0005, 2/3 wins, a positive paired series-bootstrap
95% lower bound, and positive gains in both fixed series-ID halves.

The whitening code fits AR order, EWMA scale and empirical residual CDF
only on each series' clean history and computes online prefixes causally.
Model57 tested its prefix invariance; Model58 verifies the frozen caches.
All examples and synthetic donors exclude their held and adjacent stopping
fold. The inherited ensemble and choice to pursue whitening are
development-informed; this is exploratory OOF. Sealed outer holdout and
competition test labels remain closed. No cloud training, packaging,
upload or submission. CPU only, four XGBoost threads.

Research motivation: the CrunchDAO whitening discussion
https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model58-mpl .venv/bin/python -u models/model58/run_model58.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model58-mpl .venv/bin/python -u models/model58/run_model58.py report
