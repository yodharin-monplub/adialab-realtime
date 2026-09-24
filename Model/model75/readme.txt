Model75: full-row, 500-feature GPU XGBoost voter.

Models59-64 cap every fold at 400,000 real fitting rows. Public competition
measurements and our own synthetic-data gains both indicate that the current
representation is data-limited. Model75 removes that row cap: each held-series
fold trains on every online row from the three allowed fitting folds (roughly
2.5 million real rows), plus the same 400,000 aligned synthetic rows at 40%
of training weight. The held fold and its adjacent stopping fold are excluded.

The 500 causal columns are exactly Model64's union of 439 original detector
features and 61 history-whitened features. The learner is a fixed 700-round,
depth-4, regularized histogram XGBoost model. CUDA is used only for training;
the saved trees remain compatible with CPU inference in CrunchDAO's cloud.

Pilot folds 0-1 train first. The new voter replaces 0.5 or 1.0 of Model26's
voter column, receives Model39's frozen anti-residual adjustment and causal
smoother, then contributes 0.25, 0.5, 0.75 or 1.0 to Model67. Those eight
choices are frozen before training. Only one choice selected on folds 0-1 is
evaluated on confirmation folds 2-4. Confirmation training proceeds only if
the pilot gain is at least +0.0001.

Promotion requires confirmation and full held-series OOF TSAUC >=0.640,
confirmation gain >=0.0005 over Model67, at least 2/3 confirmation wins,
positive paired-series bootstrap lower 95% bound, and positive gains in both
fixed series-ID halves. Existing feature and parent selection is exploratory;
the sealed outer holdout and competition test labels stay closed.

The Vast.ai instance must be verified, reliable, and have at least 24 GB VRAM
and 32 GB RAM. The remote Python 3.10 environment uses XGBoost 3.2.0; its UBJ
tree artifacts are readable by the newer local runtime. Preparation and reporting run locally; only the trainer runs on
the rented GPU. No instance rental, upload, or submission is automatic.

Public evidence motivating sample scale rather than another feature sweep:
https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206

Local preparation:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model75-mpl .venv/bin/python -u models/model75/prepare_model75.py

GPU training (after copying the prepared repository subset):
  python -u models/model75/train_model75_gpu.py --folds 0 1

Local pilot/final report:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model75-mpl .venv/bin/python -u models/model75/report75.py
