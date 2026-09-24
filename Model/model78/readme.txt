Model78: broader aligned-splice donor coverage.

Model77 showed that retaining more rows from the same 1,000 splice pairs is
mostly redundant. Model78 changes the other axis: it draws 1,800 distinct
prebreak donors per fold (up from 1,000), forms about 1.03 million possible
causal aligned-splice rows, and selects 400,000 of them for training. The
real row count, synthetic row count, 40% synthetic mass, 317 features,
LightGBM recipe and held/stopping exclusions remain the same as Model62.
This isolates breadth of synthetic donor combinations from sample count and
weight effects.

The new donor plans use a frozen independent seed, never repeat the prebreak
donor, and cap postbreak donor reuse at five. The new voter replaces either
one half or all of Model62's 50% allocation inside Model67; Model63 and
Model64 retain 25% each. Pilot folds 0-1 choose between those two fixed
fractions and require +0.0001 mean TSAUC. Confirmation folds 2-4 are trained
only after a pass. Promotion additionally requires full and confirmation
TSAUC >=0.640, confirmation gain >=0.0005, 2/3 wins, a positive paired-series
bootstrap lower 95% bound, and positive gains in two fixed ID halves.

All features are causal and caches are resumable/checksummed. Inherited
features, parents and the augmentation direction are development-informed,
so this remains exploratory held-series OOF. The sealed outer holdout and
competition labels stay closed. CPU only; no cloud packaging or submission.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model78-mpl timeout 9h .venv/bin/python -u models/model78/run_model78.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model78-mpl .venv/bin/python -u models/model78/run_model78.py report

