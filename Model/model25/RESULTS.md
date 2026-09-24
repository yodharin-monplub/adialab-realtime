# Model25: fixed rolling-peak strength

| Mixture | Component CV | Full ensemble CV | Gain vs Model20 | Fold wins | Pass |
|---|---:|---:|---:|---:|---|
| model20 | 0.618173 | 0.631576 | +0.000000 | 0/5 | baseline |
| model23 | 0.617708 | 0.632095 | +0.000519 | 4/5 | baseline |
| premium_05 | 0.617708 | 0.631852 | +0.000276 | 2/5 | False |
| premium_20 | 0.617708 | 0.632140 | +0.000564 | 3/5 | False |

Pass: gain >=0.0003, >=4/5 fold wins, simultaneous 95% lower bound >0 versus BOTH Model20 and Model23.
Selected: none; retain Model20.

Exploratory reused-development rolling-peak sensitivity study on Model23. Halve or double the existing correction; four simultaneous contrasts against Model20 and Model23. Conditional bootstrap excludes upstream fitting and research-selection uncertainty. No outer holdout or test labels.

These are local development scores, not new cloud scores.

Model20 remains unchanged. No packaging, upload or cloud run performed.
