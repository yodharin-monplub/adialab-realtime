Model48: time-stratified pairwise XGBoost ranker.

The official TSAUC compares positive and negative examples only at the same
online step. Model48 changes the training objective, not the causal detector
features: an XGBRanker sees each online step as a query group and optimizes
pairwise ranking. Its 400,000 real fitting rows per fold are the frozen Model20
rows. A deterministic 100,000-row sample of Model20's aligned-positive splices
adds synthetic diversity. All donors and fitting rows exclude the held series
fold and the adjacent stopping fold; all features are computed causally.

The 256 fold-specific Model20 features and most depth-3 tree parameters stay
fixed. Rank:pairwise uses mean pair sampling, two pairs per sample, 600 rounds,
and query weights proportional to positive-times-negative pair counts. This
weighting follows TSAUC's time-stratum weighting. Ranker scores are centered
and scaled on fitting rows without labels, then added to the frozen Model39
margin. Weights 0.02, 0.05, 0.10, 0.20 are screened only on pilot folds 0-1.
If no pilot weight improves their mean TSAUC by at least 0.0001, stop without
training confirmation folds. Otherwise train folds 2-4 with the selected
weight. Success requires confirmation and full OOF means >=0.640, confirmation
gain >=0.0005 over Model39, >=2/3 fold wins, positive 95% series-bootstrap
lower bound and positive gains in two prespecified disjoint series-ID halves.

The Model39 margin and feature selection are inherited and development-informed,
so this is exploratory OOF, not a fresh outer holdout. No test labels, cloud
training, packaging, upload, or submission are used automatically.

CPU only, four threads. Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model48/run_model48.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model48/run_model48.py report

Research and implementation basis:
  https://github.com/crunchdao/competitions/blob/master/competitions/structural-break-real-time/scoring/scoring.py
  https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
