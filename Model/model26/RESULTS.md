# Model26: fixed XGBoost probability mixtures

| Mixture | Component CV | Full ensemble CV | Gain vs Model23 | Fold wins | Pass |
|---|---:|---:|---:|---:|---|
| model23 | 0.617708 | 0.632095 | +0.000000 | 0/5 | baseline |
| depth_mix | 0.618445 | 0.632407 | +0.000312 | 4/5 | True |

Pass: gain >=0.0003, >=4/5 fold wins, simultaneous 95% lower bound >0 versus Model23.
Selected: depth_mix.

Exploratory reused-development depth-diversity study. One fixed candidate; conditional bootstrap excludes upstream training and selection uncertainty. Outer holdout stays closed; no test labels. Model23 remains unchanged.

These are local development scores, not new cloud scores.

Model23 remains unchanged. No packaging, upload or cloud run performed.
