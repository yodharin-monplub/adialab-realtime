# Model34: TSAUC pairwise residual correction

| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---|---|
| model26 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline | baseline |
| pairwise_residual | 0.626930 | -0.005477 | 0/5 | -0.008229 | [-0.0036914692123638515, -0.006904289612381431] | False |

Target held-series OOF TSAUC: 0.640; reached: False.

Model26 is the immutable initial margin; trees learn only a same-time pairwise additive correction.

These are local development OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
