Model79: regime-matched postbreak splice donors.

Models77-78 show that merely adding rows or random donor pairings does not
improve the aligned-splice augmentation. Model79 instead preserves every
prebreak donor A, splice time, horizon, selected synthetic row, real row,
feature, weight and learner setting from Model62's successful Model20 plan.
It changes only postbreak donor B: a deterministic nearest-regime match is
chosen from the same leakage-safe fitting pool.

Matching uses six prebreak-only descriptors over the latest 512 available
clean points: log robust scale, lag-one autocorrelation, differenced scale,
squared-value autocorrelation, log kurtosis, and low-frequency energy share.
Descriptors are robustly standardized within each fitting pool. The nearest
eligible B with sufficient postbreak tail and reuse below five is selected,
then Model20's median alignment is applied unchanged. On pilot-fold plans,
median standardized A/B distance drops from about 5.0 to 0.59 while seam
jump quantiles remain close to genuine training breaks.

The 400,000 synthetic rows retain 40% mass. The new 317-feature LightGBM
voter replaces either half or all of Model62's 50% allocation in Model67.
Folds 0-1 choose between those two frozen fractions and require +0.0001 mean
TSAUC. Only a pass trains folds 2-4. Promotion also requires full and
confirmation TSAUC >=0.640, confirmation gain >=0.0005, 2/3 wins, a positive
paired-series bootstrap lower 95% bound, and positive gains in two fixed ID
halves.

All descriptor and detector features are causal for their generated series;
matching uses training donors only and excludes held/stopping folds. The
augmentation design and parents remain development-informed, so results are
exploratory held-series OOF. The sealed outer holdout and competition labels
stay closed. CPU only; no cloud packaging, upload, or submission.

Research motivation: time-series domain adaptation benefits from aligning
temporal and frequency structure, and concatenation benchmarks explicitly
check segment compatibility:
  https://proceedings.mlr.press/v202/he23b.html
  https://link.springer.com/article/10.1007/s10618-023-00923-x

Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model79-mpl timeout 9h .venv/bin/python -u models/model79/run_model79.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model79-mpl .venv/bin/python -u models/model79/run_model79.py report

