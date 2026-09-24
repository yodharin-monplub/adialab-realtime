# Model28: pilot-selected 80% context blend

| Arm | Pilot fold 0 | Confirmation folds 1-4 | Confirmation gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model26 | 0.642736 | 0.629825 | +0.000000 | 0/4 | +0.000000 | baseline |
| pilot_80 | 0.647338 | 0.625455 | -0.004370 | 1/4 | -0.007531 | False |

Unseen confirmation target: 0.640; reached: False.
Fold 0 is excluded from confirmation because it selected the blend weight.

Model27 fold 0 is an explicitly disclosed pilot used to choose one fixed 80% new-model weight. Folds 1-4 are the untouched confirmation set for this blend decision. Training donor exclusions remain those frozen by Model27. The five-fold result is descriptive only.

These are local held-series OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
