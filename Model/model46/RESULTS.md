# Model46: CatBoost diversity voter

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | +0.000000 | 0/3 | baseline | baseline |
| CatBoost weight 0.05 | 0.638800 | 0.634260 | +0.000007 | 2/3 | -0.000034 | False |

Pilot gains by weight: {0.05: -3.060789326858604e-05, 0.1: -6.149959169166053e-05, 0.2: -0.00013207694912731327, 0.3: -0.0002192597839425492}.
Confirmation split-half gains: [1.6333809333191063e-05, -8.408377027313563e-06].
Local held-series OOF only; prior research reused development folds.

No packaging, upload, or cloud run performed.
