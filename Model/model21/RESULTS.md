# Model21: synthetic training weight

| Synthetic weight | Component CV | Full ensemble CV | Gain vs Model20 | Fold wins | Pass |
|---|---:|---:|---:|---:|---|
| aligned_25 | 0.618173 | 0.631576 | +0.000000 | 0/5 | baseline |
| aligned_40 | 0.617701 | 0.632149 | +0.000574 | 4/5 | False |
| aligned_60 | 0.613486 | 0.632129 | +0.000554 | 3/5 | False |

Pass: gain >=0.0003, >=4/5 fold wins, simultaneous 95% lower bound >0 versus Model20.
Selected: none; retain Model20.

Exploratory reused-development weight study, not independent confirmation. Same donors, rows, features, seeds and ensemble weights as Model20. Conditional bootstrap excludes prior selection/refitting uncertainty. Reserved holdout remains closed; no competition test labels.

These are local development scores, not new cloud scores.

Model20 remains unchanged. No packaging, upload or cloud run performed.
