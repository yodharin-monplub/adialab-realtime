# Model36: multi-order AR-Focus boundary features

Raw enhanced component OOF: 0.608958.

| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---|---|
| model26 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline | diagnostic |
| enhanced_05 | 0.632298 | -0.000109 | 0/5 | +0.000000 | baseline | diagnostic |
| enhanced_10 | 0.632182 | -0.000225 | 0/5 | +0.000000 | baseline | diagnostic |
| enhanced_20 | 0.631917 | -0.000489 | 0/5 | -0.000796 | [-0.00031848920374211165, -0.0007052515945136228] | False |

Target held-series OOF TSAUC: 0.640; reached: False.

Exact AR mean-shift boundary weighting across orders 1,2,5,10 appended to the frozen 256-feature backbone.

These are local development OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
