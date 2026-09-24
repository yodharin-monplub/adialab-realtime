Model96: locked confirmation of Model95's maximum scale GLR correction.

Model95 selected native additive weight 0.02 on pilot folds 0--1 with mean
gain +0.000210; the larger 0.04 weight degraded, so the maximum was internal
to the frozen grid. Model96 applies the exact locked formula to untouched
folds 2--4: candidate = Model84 + 0.02 * Model95 scale GLR.

Report full and confirmation TSAUC, fold wins, paired whole-series bootstrap
lower 95%, and two prespecified ID-half gains. Promotion requires both means
>=0.640, confirmation gain >=0.0005, 2/3 wins, lower 95% >0, and both halves
positive. No fitting, public/test labels, cloud activity, packaging, upload,
or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model96-mpl .venv/bin/python -u models/model96/run_model96.py
