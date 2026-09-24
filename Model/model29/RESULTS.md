# Model29: pilot-trained context reliability gate

| Arm | Confirmation folds 2-4 | Gain | Wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---|---|
| model26 | 0.636932 | +0.000000 | 0/3 | +0.000000 | baseline | baseline |
| context_gate | 0.635277 | -0.001655 | 0/3 | -0.003496 | [-0.0012402426794527743, -0.0017204050387426006] | False |

Unseen confirmation target: 0.640; reached: False.
Folds 0-1 train the gate and are excluded from confirmation.

One fixed linear reliability gate is trained only on pilot folds 0-1 after their Model27 predictions were observed. The gate combines Model26 and Model27 logits, time, clean-history context, and context-by-disagreement interactions. Folds 2-4 are untouched confirmation data. Five-fold results are not used as confirmatory evidence.

These are local held-series OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
