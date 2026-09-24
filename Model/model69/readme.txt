Model69: history-calibrated nonlinear rank-transition change features.

The existing supervised detector bank contains rank-crossing rates and a few
autocorrelation contrasts, but not the full conditional transition structure
between amplitude states. Model69 divides each series into four rank states
using only its clean historical quartiles. A smoothed 4x4 transition matrix is
fitted on history. At each online prefix, 64-point, 256-point and full-prefix
transition counts yield a Markov likelihood-ratio discrepancy, its centred
version, largest cell residual, and signed persistence/up/down deviations.
These 18 columns are causal; the online point at time t is the newest value
used at time t. They are computed for real and aligned-synthetic fitting rows
with the same code and prefix-parity checks.

The 317 Model62 features, exact 400,000 real and 400,000 aligned-synthetic
rows per fold, 40% synthetic mass, LightGBM recipe, donor exclusions and
five held-series folds stay fixed. Only the 18 new columns change. The four
predeclared choices are new-voter fractions 0.50 or 1.00 and Model62-voter
blend weights 0.50 or 1.00 in the frozen Model39 stack. Pilot folds 0-1 must
gain at least +0.0001 to justify confirmation fits. Promotion needs both
full and confirmation TSAUC >=0.640, confirmation gain >=+0.0005, at least
2/3 confirmation wins, positive paired-series bootstrap lower 95% bound,
and gains in both fixed ID halves. These are exploratory held-series OOF
scores: inherited features and stack were tuned on development data. The
sealed outer holdout and competition test labels remain closed.

CPU only (four LightGBM threads); checkpointed and resumable. No automatic
packaging, upload or cloud submission. Stop after 15 hours of local runtime.
The earlier challenge's published solutions motivate complexity/shape signals,
but their offline task differs materially from this real-time protocol:
https://github.com/gsoisson/adia-structural-break
https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge

Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model69-mpl .venv/bin/python -u models/model69/run_model69.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model69-mpl .venv/bin/python -u models/model69/run_model69.py report
