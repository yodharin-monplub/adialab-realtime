# Model64: full-detector 40%-synthetic LightGBM voter

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | baseline | baseline | baseline | baseline |
| all 500 features, old fraction 0.50, voter weight 1.00 | 0.640528 | 0.637508 | +0.001735 | 2/3 | -0.001444 | False |

Pilot gains: {'(0.0, 0.5)': 0.0029486631813806508, '(0.0, 1.0)': 0.0038984019567925854, '(0.5, 0.5)': 0.003696402061326598, '(0.5, 1.0)': 0.005497483044651252}.
Confirmation split-half gains: [0.0037033270987553157, 8.468502430780678e-05].
Exploratory held-series OOF; inherited features and ensemble are development-informed.

No packaging, upload, or cloud run performed.
