Model94: locked confirmation of Model93's direct scale correction.

Model93 selected weight 0.04 on pilot folds 0--1 with mean gain +0.000515.
Model94 freezes that exact native-score formula and evaluates untouched folds
2--4. Candidate = Model84 score + 0.04 * Model90 causal scale log-Bayes
evidence. No new weight, feature, learner, or postprocessing choice is made.

Report the five fold scores, confirmation and full means, fold wins, paired
whole-series bootstrap lower 95%, and gains in two prespecified series-ID
halves. Promotion requires confirmation and full TSAUC >=0.640, confirmation
gain >=0.0005, 2/3 wins, lower 95% >0, and both split-half gains >0.

This is label-free postprocessing of already-frozen OOF signals. No model
fitting, public/test labels, cloud activity, packaging, upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model94-mpl .venv/bin/python -u models/model94/run_model94.py
