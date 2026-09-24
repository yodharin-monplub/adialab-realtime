Model102: train the scale GLR as a conditional LightGBM feature.

Model96 established that maximum-over-tau scale evidence has complementary OOF
signal, but its globally additive correction cannot adapt to elapsed online
time or interactions with the incumbent detector bank. Model102 appends the
single causal scale-GLR column to Model84's 317-column feature contract and
re-trains the exact frozen expressive LightGBM recipe.

Real and synthetic feature rows use the identical history-only whitening and
GLR calculation. Original real data remains combined with the frozen 40%
synthetic TSAUC cell mass; folds, donor plans, row counts, fitting weights,
learner settings, postprocessing, and Model67 outer ensemble remain unchanged.
The implementation reuses Model88's audited feature-extension engine while
binding it to this directory and feature definition.

Pilot folds 0-1 select blend weight 0.25, 0.50, or 0.75 against Model84 and
require +0.0001 before confirmation folds train. CPU only and resumable; peak
RAM is expected near 14 GiB and each fold approximately 4-5 minutes. No cloud
activity, packaging, upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model102-mpl .venv/bin/python -u models/model102/run_model88.py
