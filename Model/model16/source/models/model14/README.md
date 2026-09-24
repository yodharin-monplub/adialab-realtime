# Model 14: time-varying pairwise stacking

Model 14 extends Model 9's 34 primitive predictors with Model 11's compact
multiscale predictor. A logistic model is fitted to 200,000 mirrored pairs per
fold, with positives and negatives matched at the same online time. Pair time
strata are sampled in proportion to their positive-negative pair counts.

The fixed time basis is `[1, 0.35 exp(-t/32), 0.35 exp(-t/128)]`. Pair differences
are scaled using fit pairs only. Interaction columns reuse the static scales,
so their factor of 0.35 imposes stronger shrinkage on deviations from the static
weights. The regularization is C=0.001, with no held-fold parameter search.

Run from the project root after Model 11 completes:

```bash
OPENBLAS_NUM_THREADS=1 python3 models/model14/run_model14.py
```

All input OOF vectors are checked against saved hashes, dimensions and ranges.
The original Model 5 complete-series folds are retained. Each meta-held fold is
excluded from pair sampling and feature scaling. Source-hashed runs retain
coefficients, scales, fold predictions and score comparisons. No outer holdout
or competition test labels are read.

The 34 original base models are not refitted inside each meta-held fold. Their
training and selection reused these development folds, and Model 11 was chosen
after inspecting its diversity result. Scores therefore remain exploratory.
There is no automatic deployment, prize claim, or leaderboard-score claim.
Full base refitting and streaming integration are required before promotion.
