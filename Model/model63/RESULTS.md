# Model63: extra-randomized 40%-synthetic LightGBM voter

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | baseline | baseline | baseline | baseline |
| extra-trees 40%, old fraction 0.50, voter weight 1.00 | 0.640260 | 0.637420 | +0.001467 | 2/3 | -0.001919 | False |

Pilot gains: {'(0.0, 0.5)': 0.002820639500714428, '(0.0, 1.0)': 0.0040895215337346524, '(0.5, 0.5)': 0.0036520284717471707, '(0.5, 1.0)': 0.005678940244616237}.
Confirmation split-half gains: [0.0027545751568580634, 0.0005415554752129514].
Exploratory held-series OOF; inherited features and ensemble are development-informed.

No packaging, upload, or cloud run performed.
