Model99: locked combined Discord scale evidence.

Model98 selected, on pilot folds 0-1 only, the combination Model84 native
score + 0.02 times maximum-over-tau Gaussian scale GLR + 0.04 times uniformly
tau-integrated inverse-gamma scale Bayes evidence. Model99 freezes that formula
and evaluates confirmation folds 2-4 without any additional selection.

Report full and confirmation TSAUC, per-fold gains against Model84, residual
gains against Model96's GLR-only correction, a paired series bootstrap on the
confirmation folds, and two fixed series-ID halves. Promotion still requires
full OOF >= 0.64, confirmation gain >= 0.0005, at least 2/3 confirmation-fold
wins, positive lower 95% bootstrap bound, and positive gains in both halves.
CPU only; no cloud packaging, upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model99/run_model99.py
