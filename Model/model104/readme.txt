Model104: nested cross-fitted empirical-null calibration of scale GLR.

Sequential scan-statistic research shows that fixed/asymptotic calibration can
produce time-dependent false-positive rates. Model104 converts Model95's causal
maximum scale GLR into a time-specific empirical tail score. For each exact
observable online step, it estimates the null CDF from target-zero rows in
fitting folds and emits -log(P_null(GLR >= observed)), clipped at 12. Mid-ranks
handle ties. The transform is monotone within time, so it preserves standalone
GLR TSAUC while calibrating its spacing relative to Model84.

The evaluation is nested and series-fold clean. For outer held fold h, its null
tables use only the other four folds. A correction weight from
{0, .005, .01, .02, .04, .08} is selected by four inner held-fold scores; each
inner fold j is itself transformed using only the three folds excluding h and
j. Thus neither held labels nor held null examples affect h's transform or
weight. At deployment, null tables can be fit once on all available training
rows.

Promotion requires full OOF >= .64, gain >= .0005 over Model84, at least 3/5
wins, positive paired-series bootstrap lower 95%, and positive gains in both
fixed ID halves. CPU only; no cloud activity, packaging, upload, or submission.

Research motivation:
  https://proceedings.mlr.press/v151/cobb22a.html
  https://proceedings.mlr.press/v98/maillard19a.html
  https://arxiv.org/abs/1309.5264

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model104-mpl .venv/bin/python -u models/model104/run_model104.py
