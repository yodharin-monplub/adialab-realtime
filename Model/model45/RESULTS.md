# Model45: multiscale kernel discrepancy

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | +0.000000 | 0/3 | baseline | baseline |
| kernel weight 0.05 | 0.638895 | 0.634183 | +0.000102 | 2/3 | -0.000587 | False |

Pilot gains by weight: {0.05: -0.0003648348215685715, 0.1: -0.0009346920923478663, 0.2: -0.002647122952961656, 0.4: -0.00740162974213554}.
Confirmation split-half gains: [2.379327379287144e-05, 0.0001621641796566431].
These are local held-series OOF scores; prior research reused development folds.

No packaging, upload, or cloud run performed.
