# Model32: five-fold cross-fitted diversity stack

| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---|---|
| model26 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline | baseline |
| crossfit_stack | 0.632158 | -0.000249 | 3/5 | -0.001599 | [-0.00046612213328978136, -0.0008889647158247804] | False |

Target held-series OOF TSAUC: 0.640; reached: False.

Every meta prediction is produced by a model fitted and standardized on the other four OOF folds.

These are local OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
