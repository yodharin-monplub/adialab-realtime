Model92: direct scale log-Bayes evidence correction.

The Discord discussion reported that one evidence column could help while a
larger bundle diluted it. Model90 confirmed that its scale feature improves
both raw pilot components, but Model90 and Model91 found only about +0.00006
after refitting/blending. Model92 tests the remaining minimal interpretation:
add the scale log Bayes factor directly to Model84's safe-clipped logit.

Weights 0.005, 0.01, 0.02, 0.04 and 0.08 are frozen before scoring. The
evidence is the exact Model90 causal feature: Model57 history whitening,
inverse-gamma amplitude mixture, and uniform integration over every candidate
tau represented by the geometric grid cells. TSAUC compares rankings only
within equal online ages, so age-dependent offsets require no cross-series
calibration. Each prediction remains independently deployable.

Only pilot folds 0--1 are evaluated. A +0.0001 gain is required before any
confirmation study. No fitting, public/test labels, cloud work, packaging,
upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model92-mpl .venv/bin/python -u models/model92/run_model92.py
