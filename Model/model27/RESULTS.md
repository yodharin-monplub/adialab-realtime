# Model27: family-matched contrastive augmentation

| Arm | Component CV | Full ensemble CV | Gain | Fold wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---:|---|---|
| model26 | 0.618445 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline | baseline |
| matched_context | 0.621530 | 0.632335 | -0.000072 | 3/5 | -0.000713 | [-0.00032420891903783833, 0.00026546835859031503] | False |

Target held-series OOF TS-AUC: 0.640; reached: False.
Promotion requires gain >=0.0007, >=4/5 folds, lower bound >0, and positive gains in both prespecified disjoint series halves.

Prespecified family-matched contrastive augmentation. Every new fit excludes its held series and the adjacent stopping fold. Positive and hard-negative composites share seam construction. History context is inference-safe. Existing feature selection and ensemble weights remain development-informed; outer holdout and test labels remain closed.

These are local OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
