Model66: higher-lag dependence evidence on history-fitted innovations.

Model57 fits an AR(p<=12), EWMA conditional scale and empirical normal-score
transform on each series' guaranteed clean history, then uses the resulting
online innovations in moment, tail and lag-one channels. Model66 adds 40
causal features: signed and absolute innovation products at lags 2, 4, 8
and 16, centered and scaled by clean-history products and accumulated over
16, 64, 256 and full-observed-prefix windows, plus a maximum-absolute
summary per channel. Each value at time t uses history and online data
only through t. Prefix parity is checked on real and synthetic series.

The same 400,000 real and 400,000 aligned-synthetic fitting rows per fold,
40% synthetic mass, donor exclusions, 317 original-plus-whitened features,
standard LightGBM recipe and Model39 stack from Model62 are retained. Only
the 40 innovation-dependence features are added. Real and synthetic feature
caches are checkpointed with block digests. CPU only, four fit threads.

The new voter alone or a 50/50 average with frozen Model62 replaces 0.5 or
1.0 of Model26's voter. Four choices are frozen before training. Pilot folds
0-1 select one by mean gain over Model62; a gain <+0.0001 stops before
confirmation folds 2-4. Promotion requires confirmation and full five-fold
TSAUC >=0.640, confirmation gain >=0.0005 over Model62, 2/3 wins, positive
paired-series bootstrap lower 95% bound, and positive gains in both fixed
series-ID halves. Inherited features and stack make this exploratory OOF;
the sealed outer holdout and competition test labels remain closed. No cloud
packaging, upload or submission is automatic.

This tests a distinct conditional-dependence representation after raw and
differenced lag contrasts in Model55 failed their pilot. Online change in
autoregressive dynamics is a documented change-point problem, but that
literature does not establish performance for these features here:
https://arxiv.org/abs/0708.2369

Run/resume from repository root, within the 15-hour limit:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model66-mpl timeout 14h .venv/bin/python -u models/model66/run_model66.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model66-mpl .venv/bin/python -u models/model66/run_model66.py report
