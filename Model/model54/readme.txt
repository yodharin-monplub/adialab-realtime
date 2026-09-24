Model54: anti-reselected-voter experiment.

Model53's late-horizon Fisher-reselected XGBoost voter lost TSAUC at every
positive pilot blend weight. That does not prove the opposite direction helps,
but its difference from the established Model26 voter could expose a
systematic error direction, as the anti-residual Model35 did. Model54 tests a
negative coefficient on exactly that difference before the frozen Model39
stack and causal smoother. This is an exploratory, pilot/confirmation test,
not a claim that an inferior voter is a superior standalone model.

The fixed blend coefficients are -0.10, -0.20, -0.40 and -0.80. The two
completed Model53 pilot-fold predictions are read with checkpoint checksums.
Pilot folds 0-1 select one coefficient. If pilot mean gain over Model39 is
below +0.0001, stop. Otherwise the exact same Model53 400,000-real plus
400,000-aligned-synthetic fit recipe runs on confirmation folds 2-4, with the
resulting checkpoints isolated under Model54. These folds were not trained or
inspected for the Model53 voter. The candidate's probability is clipped to
[1e-6, 1-1e-6] before the existing stack.

Promotion requires confirmation and full held-series OOF TSAUC >=0.640,
confirmation gain >=0.0005, at least 2/3 fold wins, positive 95% paired
series-bootstrap lower bound and positive gains in both prespecified series-ID
halves. The inherited Model39 stack, feature union and pilot direction are
development-informed, so all scores are exploratory. The sealed outer
holdout and competition test labels remain closed; no automatic cloud run or
submission. CPU only, four XGBoost threads.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model54-mpl .venv/bin/python -u models/model54/run_model54.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model54-mpl .venv/bin/python -u models/model54/run_model54.py report
