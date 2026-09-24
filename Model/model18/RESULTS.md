# Model 18: training-only splice augmentation

Five whole-series development folds; fresh Model 11 control. Higher TSAUC is better.

| Arm | Mean TSAUC | Gain | Fold wins | Simultaneous 95% lower gain |
|---|---:|---:|---:|---:|
| control | 0.570593 | +0.000000 | 0/5 | — |
| raw_10 | 0.575496 | +0.004904 | 4/5 | -0.001821 |
| aligned_10 | 0.574869 | +0.004276 | 4/5 | -0.002449 |
| raw_25 | 0.574856 | +0.004264 | 4/5 | -0.002461 |
| aligned_25 | 0.575140 | +0.004548 | 4/5 | -0.002177 |

Pass requires gain >= 0.001, at least four fold wins, and simultaneous lower bound > 0.
Selected standalone candidate: none; augmentation did not pass the fixed gates.

## Fold TSAUC

| Arm | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| control | 0.583322 | 0.550736 | 0.576362 | 0.568840 | 0.573703 |
| raw_10 | 0.590771 | 0.564153 | 0.578939 | 0.579663 | 0.563955 |
| aligned_10 | 0.589705 | 0.561770 | 0.580865 | 0.574475 | 0.567530 |
| raw_25 | 0.592054 | 0.559257 | 0.584460 | 0.570948 | 0.567562 |
| aligned_25 | 0.591848 | 0.559484 | 0.585218 | 0.574562 | 0.564589 |

## Real no-break false alarms

Fraction of complete real no-break series with any online prediction >= the fixed threshold; not calibrated probabilities.

| Arm | >= 0.5 | >= 0.8 |
|---|---:|---:|
| control | 81.35% | 1.88% |
| raw_10 | 83.12% | 3.91% |
| aligned_10 | 83.76% | 3.79% |
| raw_25 | 83.81% | 8.32% |
| aligned_25 | 84.02% | 8.18% |

Reused development folds only. Paired whole-series bootstrap conditions on saved OOF predictions, not upstream research or training uncertainty. Synthetic labels are assumptions, not guaranteed detectable breaks.

The historically revealed outer holdout remained closed. No competition test labels were used.

Model 16 unchanged; no ensemble, cloud submission, or full-data promotion.

Run artifacts: `models/model18/runs/ae6d2a6154348e82`.
Full paired fold gains, donor shortfalls, weight accounting, and counts are in `summary.json`.
