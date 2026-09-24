# Model37: independent-seed depth-voter bag

| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---|---|
| model26 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline | diagnostic |
| replica | 0.632286 | -0.000121 | 2/5 | +0.000000 | baseline | diagnostic |
| seed_bag | 0.632359 | -0.000048 | 2/5 | -0.000111 | [2.4273454924617077e-05, -0.00011316444267528425] | False |

Target held-series OOF TSAUC: 0.640; reached: False.

Independent XGBoost seed bag; all data, features, folds, weights and hyperparameters fixed.

These are local development OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
