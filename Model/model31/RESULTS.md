# Model31: direct pilot-trained meta-stack

| Arm | Pilot fold0 | Confirmation folds1-4 | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model26 | 0.642736 | 0.629825 | +0.000000 | 0/4 | +0.000000 | baseline |
| direct_meta | 0.644463 | 0.630030 | +0.000205 | 1/4 | -0.001141 | False |

Unseen confirmation target: 0.640; reached: False.
Fold0 trains the meta-model and is excluded from confirmation.

Method prespecified in prespec.json after Model30 fold0 only; folds1-4 are confirmation.

These are local held-series OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
