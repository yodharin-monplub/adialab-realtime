Model93: native-scale direct Bayesian scale-evidence correction.

Model92 incorrectly treated Model84's final score as a probability, clipped
values outside [0,1], and destroyed the incumbent ordering. Model93 preserves
that failed run and repairs only the representation: candidate = Model84
native score + weight * Model90 scale log Bayes evidence.

Before scoring, a label-free scale audit measured pilot standard deviations
0.571 for the Model84 native score and 1.021 for the evidence. Weights 0.0025,
0.005, 0.01, 0.02 and 0.04 were then frozen. The zero-weight identity is
verified against Model84's exact five fold scores.

This remains the Discord-motivated individual-evidence-column ablation. Only
pilot folds 0--1 are evaluated; a +0.0001 gain is required before any separate
confirmation model. No fitting, public/test labels, cloud activity, packaging,
upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model93-mpl .venv/bin/python -u models/model93/run_model93.py
