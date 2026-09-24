Model60: larger replacement weights for Model59's verified LightGBM voter.

Model59's frozen pilot gain increased monotonically through the largest tested
weight, 0.40, and its full five-fold OOF improved to 0.636217. This study
tests the prespecified higher weights 0.50, 0.60, 0.80 and 1.00 without
retraining or touching features. The Model59 checkpoints are read only and
verified; its probabilities replace the same Model26 voter column in the
unchanged Model39 ensemble and causal smoother. Model59's chosen 0.40 blend
is a reference, not part of the new selection grid.

Pilot folds 0-1 select exactly one of the four new weights. A gain below
+0.0001 versus Model39 stops the study. Confirmation folds 2-4 then require
confirmation and full five-fold TSAUC >=0.640, gain >=0.0005, 2/3 wins,
positive paired-series bootstrap lower 95% bound and positive gains in both
fixed ID halves. All scores are exploratory development held-series OOF;
the sealed outer holdout and competition test labels stay closed. No cloud
packaging, upload, submission or training.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model60-mpl .venv/bin/python -u models/model60/report60.py
