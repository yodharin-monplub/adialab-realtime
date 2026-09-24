Model101: locked maximum-over-tau dependence GLR correction.

Model100 selected additive weight 0.02 using pilot folds 0-1 only. Model101
freezes Model84 native score + 0.02 times the causal two-sided dependence GLR
and evaluates confirmation folds 2-4 without further selection.

Report full and confirmation TSAUC, per-fold gains, a paired series bootstrap
on the confirmation folds, and two fixed series-ID halves. Promotion requires
full OOF >= 0.64, confirmation gain >= 0.0005, at least 2/3 confirmation-fold
wins, a positive lower 95% bootstrap bound, and positive gains in both halves.
CPU only; no cloud packaging, upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model101-mpl .venv/bin/python -u models/model101/run_model101.py
