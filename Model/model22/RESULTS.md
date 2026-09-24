# Model22: all real data plus synthetic rows

| Training data | Component CV | Full ensemble CV | Gain vs Model20 | Fold wins | Pass |
|---|---:|---:|---:|---:|---|
| model20 | 0.618173 | 0.631576 | +0.000000 | 0/5 | baseline |
| all_real | 0.610304 | 0.628879 | -0.002696 | 0/5 | False |
| all_real_plus_synthetic | 0.616467 | 0.630938 | -0.000638 | 1/5 | False |

Pass: gain >=0.0003, >=4/5 fold wins, simultaneous 95% lower bound >0 versus Model20; augmented must also pass versus all-real.
Selected: none; retain Model20.

Exploratory reused-development experiment. All real fitting rows retained. Synthetic rows appended without reducing real weights. Same donors, features, seeds and ensemble settings as Model20. Conditional bootstrap excludes prior selection/refitting uncertainty. Reserved outer holdout stays closed; no competition test labels.

These are local development scores, not new cloud scores.

Model20 remains unchanged. No packaging, upload or cloud run performed.
