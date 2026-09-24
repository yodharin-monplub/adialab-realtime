# Model62: 40%-synthetic LightGBM voter

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | baseline | baseline | baseline | baseline |
| 40%-mass, old fraction 0.00, voter weight 1.00 | 0.639785 | 0.637534 | +0.000992 | 2/3 | -0.002708 | False |

Pilot gains: {'(0.0, 0.25)': 0.0024304893231805735, '(0.0, 0.5)': 0.00432656532038922, '(0.0, 0.75)': 0.005741665833699727, '(0.0, 1.0)': 0.006677140248089408, '(0.5, 0.25)': 0.0022015785268616472, '(0.5, 0.5)': 0.003932019049651458, '(0.5, 0.75)': 0.005218838865228337, '(0.5, 1.0)': 0.006060742555778265}.
Confirmation split-half gains: [0.0024967624536229582, -0.0002260776364788608].
Exploratory held-series OOF; inherited features and ensemble are development-informed.

No packaging, upload, or cloud run performed.
