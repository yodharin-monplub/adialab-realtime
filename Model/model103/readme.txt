Model103: cross-fitted early/late calibration of maximum scale GLR.

Model96 showed that a global 0.02 scale-GLR correction improves four of five
folds, while research on sequential scan statistics warns that fixed calibration
can be inaccurate across monitoring time. Model103 tests the smallest useful
time-varying extension: one weight for observable online steps 1-32 and one for
steps 33 onward. This uses elapsed online time, never the unknown break age.

Weights are selected from {0, 0.005, 0.01, 0.02, 0.04}. For each held fold,
the early/late pair maximizing mean TSAUC on the other four complete held-series
folds is frozen, then applied to the held fold. Thus every Model103 row is
predicted with calibration that never used its fold's labels. The 25 candidate
pairs and the cutoff are fixed before scoring. This is a cross-fitted model, not
a pilot/confirmation reuse.

Promotion requires full OOF >= 0.64, gain >= 0.0005 over Model84, at least 3/5
fold wins, a positive paired-series bootstrap lower 95% bound, and positive
gains in two fixed series-ID halves. CPU only; no cloud activity, packaging,
upload, or submission.

Research motivation:
  https://proceedings.mlr.press/v151/cobb22a.html
  https://proceedings.mlr.press/v98/maillard19a.html
  https://arxiv.org/abs/1309.5264

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model103-mpl .venv/bin/python -u models/model103/run_model103.py
