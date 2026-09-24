# Model103: cross-fitted early/late scale-GLR calibration

| Arm | Full OOF TSAUC | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---|
| Model84 | 0.637888 | baseline | baseline | baseline | baseline |
| Cross-fitted early/late GLR | 0.638204 | +0.000316 | 4/5 | -0.000098 | False |

Selected weights by held fold: [{'held_fold': 0, 'selected_index': 23, 'early_weight': 0.04, 'late_weight': 0.02, 'training_fold_mean': 0.6349593382363229, 'training_gain': 0.0004095964295867116}, {'held_fold': 1, 'selected_index': 23, 'early_weight': 0.04, 'late_weight': 0.02, 'training_fold_mean': 0.6434909433249227, 'training_gain': 0.0002638678757891366}, {'held_fold': 2, 'selected_index': 23, 'early_weight': 0.04, 'late_weight': 0.02, 'training_fold_mean': 0.6366562554332733, 'training_gain': 0.00029611362764125904}, {'held_fold': 3, 'selected_index': 23, 'early_weight': 0.04, 'late_weight': 0.02, 'training_fold_mean': 0.6417216821566682, 'training_gain': 0.0002838337300792304}, {'held_fold': 4, 'selected_index': 23, 'early_weight': 0.04, 'late_weight': 0.02, 'training_fold_mean': 0.6341912927912812, 'training_gain': 0.00032677266972136465}].

Fold gains: [-5.8201385529255134e-05, 0.0005247128296610448, 0.00039572982225244413, 0.00044484941250066967, 0.0002730936539325768]. Split-half gains: [0.00011994258560210636, 0.0004925316637617305].

No packaging, upload, or cloud run performed.
