# Model61: extra-randomized LightGBM voter

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | baseline | baseline | baseline | baseline |
| extra-trees, old fraction 0.50, voter weight 1.00 | 0.640179 | 0.636831 | +0.001386 | 2/3 | -0.001499 | False |

Pilot gains: {'(0.0, 0.25)': 0.001292153500418225, '(0.0, 0.5)': 0.002190175888079793, '(0.0, 0.75)': 0.0027571854774433024, '(0.0, 1.0)': 0.0030009093695438915, '(0.5, 0.25)': 0.0016389765760576047, '(0.5, 0.5)': 0.002890810134374311, '(0.5, 0.75)': 0.003768431064353317, '(0.5, 1.0)': 0.004327853786806557}.
Confirmation split-half gains: [0.00231629057637861, 0.0008271786797926506].
Exploratory held-series OOF; inherited features and ensemble are development-informed.

No packaging, upload, or cloud run performed.
