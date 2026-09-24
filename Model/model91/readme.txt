Model91: pre-stack blend of Model84 and Model90 scale components.

Model90's scale-only LightGBM component improved both pilot folds substantially
(+0.00320 and +0.00421), yet blending its fully postprocessed signal over the
entire Model84 final stack gained only +0.000059. Model91 tests whether the
useful scale information is being attenuated by that topology.

For component weights 0.125, 0.25, 0.50, 0.75 and 1.0, Model91 first blends
the Model84 expressive probability and Model90 scale probability. It then
applies the unchanged ensemble-margin replacement, anti-delta correction, and
causal smoother exactly once. Model84's fixed 0.50 outer blend with Model67 is
preserved. Weight zero reconstructs Model84 exactly.

Only already-open pilot folds 0--1 are scored. A +0.0001 mean gain is required
before a separately frozen confirmation model may train folds 2--4. No labels,
model fitting, public data, cloud activity, packaging, upload, or submission
occur here. This is a topology ablation over immutable saved OOF predictions.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model91-mpl .venv/bin/python -u models/model91/run_model91.py
