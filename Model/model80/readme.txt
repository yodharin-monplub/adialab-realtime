Model80: TSAUC-mass-aligned sampling of the original splice pool.

Model20 creates roughly 575,000 causal rows from 1,000 aligned real-to-real
splice pairs, then uniformly samples 400,000. Model80 keeps those exact pairs,
their order, all real fitting rows, the 317 features, 40% synthetic mass and
Model62 LightGBM recipe. Only the 400,000 retained synthetic indices change.

For each fitting fold, the official TSAUC pair mass at online time t is
n_positive(t)*n_negative(t), and both label sides of that stratum receive the
same desired representation. Each possible synthetic row is weighted by its
fitting-fold pair mass divided by available synthetic rows in its
(time_online, label) cell, then 400,000 rows are drawn without replacement by
a frozen seed. Held and stopping fold labels are excluded. On pilot plans,
correlation between retained cell counts and fitting pair mass rises from
about 0.57 to 0.89, while count-per-mass variability falls from 5-6x to 0.50x.
Mass-preserving training weights remain unchanged; the resampling only lowers
representation noise where TSAUC places more weight.

The resulting voter replaces either half or all of Model62's 50% allocation
inside Model67. Folds 0-1 choose between those two frozen fractions and must
gain +0.0001 mean TSAUC. Only a pass trains folds 2-4. Promotion also requires
full and confirmation TSAUC >=0.640, confirmation gain >=0.0005, 2/3 wins, a
positive paired-series bootstrap lower 95% bound, and positive gains in two
fixed ID halves.

All detector features are causal and all donors exclude held/stopping folds.
The resampling uses labels only from the fitting series. Inherited features,
parents and augmentation direction remain development-informed, so results
are exploratory held-series OOF. The sealed outer holdout and competition
labels stay closed. CPU only; no cloud packaging, upload, or submission.

Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model80-mpl timeout 9h .venv/bin/python -u models/model80/run_model80.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model80-mpl .venv/bin/python -u models/model80/run_model80.py report

