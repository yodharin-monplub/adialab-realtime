Model83: pilot-selected causal persistence correction over Model67.

Model67 averages three separately postprocessed LightGBM-based OOF signals.
That averaging can reintroduce small one-step rank reversals even though a true
break is absorbing. Model83 tests a frozen grid of deployable, strictly causal
secondary transforms: short EMA shrinkage, rolling maximum premiums, and
decayed-peak premiums. The raw Model67 ranking is included as the null arm.

Only folds 0-1 select the transform. Selection must improve their mean TSAUC by
at least 0.0001 or the raw arm is retained. Folds 2-4 are then evaluated once.
Promotion requires confirmation and full OOF TSAUC >=0.640, confirmation gain
>=0.0005, at least 2/3 confirmation-fold wins, a positive paired-series
bootstrap lower 95% bound, and positive gains in both fixed ID halves.

All operations use only a series' current and past Model67 scores. No raw-data
lookahead, cross-series state, cloud run, packaging, or submission is involved.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model83/run_model83.py
