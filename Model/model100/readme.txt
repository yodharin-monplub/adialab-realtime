Model100: maximum-over-tau dependence GLR on whitened innovations.

The Discord brief reports that scale and dependence were the useful Bayesian
channels. Model90 tested dependence only after uniform averaging over candidate
break times. Model100 instead mirrors the more successful Model95 strategy:
for each causal suffix of length 6,8,12,...,768 and the full observed prefix,
compute the two-sided Gaussian correlation generalized likelihood ratio and
take its maximum over candidate suffixes.

For a suffix, rho is the no-intercept correlation of consecutive whitened
innovations and the evidence is -0.5*(n-1)*log(1-rho^2). It detects either
positive or negative serial-dependence changes. Lengths below six are suppressed
to reduce extreme three/four-point correlations. The same Model57 history-only
whitening is reused, and prefix parity is checked on three real series.

After a label-free cache audit, pilot folds 0-1 select one native-score additive
weight from 0.00125, 0.0025, 0.005, 0.01, 0.02, and 0.04. Confirmation remains
closed unless pilot gain is at least 0.0001. CPU only; no cloud activity,
packaging, upload, or submission.

Run from repository root:
  MPLCONFIGDIR=/tmp/model100-mpl .venv/bin/python -u models/model100/build_glr.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model100-mpl .venv/bin/python -u models/model100/run_model100.py
