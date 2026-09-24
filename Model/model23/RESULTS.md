# Model23: fixed XGBoost probability mixtures

| Mixture | Component CV | Full ensemble CV | Gain vs Model20 | Fold wins | Pass |
|---|---:|---:|---:|---:|---|
| model20 | 0.618173 | 0.631576 | +0.000000 | 0/5 | baseline |
| mix_40 | 0.618884 | 0.631947 | +0.000372 | 4/5 | False |
| mix_60 | 0.617708 | 0.632095 | +0.000519 | 4/5 | True |

Pass: gain >=0.0003, >=4/5 fold wins, simultaneous 95% lower bound >0 versus Model20.
Selected: mix_60.

Exploratory follow-up on reused development folds, not independent confirmation. Two fixed equal-probability mixtures: Model20 aligned25 XGBoost with Model21 aligned40 or aligned60. No weight search, new fitting or change to other ensemble members. Conditional bootstrap excludes upstream fitting and selection uncertainty. Reserved holdout stays closed; no competition test labels. Model22 continues independently and is not used to select these mixtures.

These are local development scores, not new cloud scores.

Model20 remains unchanged. No packaging, upload or cloud run performed.
