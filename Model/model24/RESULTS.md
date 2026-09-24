# Model24: training-size diversity mixtures

| Mixture | Component CV | Full ensemble CV | Gain vs Model20 | Fold wins | Pass |
|---|---:|---:|---:|---:|---|
| model20 | 0.618173 | 0.631576 | +0.000000 | 0/5 | baseline |
| model23 | 0.617708 | 0.632095 | +0.000519 | 4/5 | baseline |
| mix_25 | 0.618594 | 0.631478 | -0.000098 | 1/5 | False |
| mix_50 | 0.618445 | 0.631339 | -0.000237 | 1/5 | False |

Pass: gain >=0.0003, >=4/5 fold wins, simultaneous 95% lower bound >0 versus BOTH Model20 and Model23.
Selected: none; retain Model20.

Exploratory reused-development follow-up after Model22 standalone results. Two fixed training-size probability mixtures; four simultaneous contrasts against Model20 and Model23. Conditional bootstrap excludes upstream fitting and research-selection uncertainty. No outer holdout or test labels.

These are local development scores, not new cloud scores.

Model20 remains unchanged. No packaging, upload or cloud run performed.
