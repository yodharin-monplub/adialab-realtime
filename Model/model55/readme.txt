Model55: causal higher-lag dependence-shift features.

The original 439-feature detector union contains detailed moment, tail and
scale evidence, but little explicit serial-dependence information beyond lag
two. A change in autoregressive or oscillatory dynamics can alter several
higher lags while leaving location and variance nearly unchanged. Model55
adds lag-correlation contrast statistics at lags 2, 4, 8 and 16 on two
history-standardized streams: the raw series and its first differences.
For each lag, online correlations are computed over windows 64, 256 and the
entire observed prefix and compared with the corresponding clean-history
correlation using a Fisher-z difference scaled by sqrt(number of pairs).
Max-absolute contrasts across lags provide six additional omnibus features.
All 30 columns use only the clean history and currently observed online
prefix. They do not use the final online length, future observations, a
candidate series' label, or any other test series. The implementation tests
prefix invariance on representative series before training.

The 30 columns are appended to Model20's original 256 fold-specific causal
features. The same 400,000 real fitting rows, 400,000 aligned synthetic rows,
25% synthetic mass share, depth-3/800-round XGBoost recipe and Model39 stack
are used. Original Model20 real/synthetic feature caches are verified and
reused read-only. No held or adjacent stopping fold enters training or
synthetic donors. Pilot folds 0-1 select a weight from 0.05, 0.10, 0.20,
0.40. If pilot gain is below +0.0001, stop before folds 2-4. Promotion
requires confirmation and five-fold TSAUC >=0.640, confirmation gain
>=0.0005, at least 2/3 wins, positive paired series-bootstrap lower bound
and positive gains in both fixed series-ID halves.

This is exploratory held-series OOF: inherited feature selection and stack
are development-informed. The sealed outer holdout and competition test
labels remain closed. No cloud training, packaging, upload or submission.
CPU-only, four XGBoost threads.

Research motivation: changes in spectral density / higher-order dependence
can signal regime changes not captured by mean and scale shifts; see
Mosaferi, Safikhani & Bai (2025), https://arxiv.org/abs/2503.23211.
This experiment is not an implementation of their change-point estimator.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model55-mpl .venv/bin/python -u models/model55/run_model55.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model55-mpl .venv/bin/python -u models/model55/run_model55.py report
