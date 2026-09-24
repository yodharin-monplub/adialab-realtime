# Model33: nonlinear primitive OOF stack

| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---|---|
| model26 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline | baseline |
| nonlinear_stack | 0.620470 | -0.011937 | 0/5 | -0.017130 | [-0.008507875777642115, -0.015807201257925008] | False |

Target held-series OOF TSAUC: 0.640; reached: False.

Shallow nonlinear meta-model over frozen primitive OOF logits. Held and adjacent stopping folds are excluded from each fit; the documented Model9 base-level nesting caveat still applies.

These are local development OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
