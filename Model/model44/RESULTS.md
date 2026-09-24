# Model44: regularized LightGBM diversity voter

| Arm | Confirmation TSAUC | Gain | Wins | Lower 95% | Split gains | Pass |
|---|---:|---:|---:|---:|---|---|
| model39 | 0.638793 | +0.000000 | 0/3 | +0.000000 | baseline | baseline |
| LightGBM weight 0.30 | 0.638852 | +0.000059 | 2/3 | -0.000191 | [5.1658059999848405e-05, 0.00010272876496664196] | False |

Pilot gains by weight: {0.05: 1.81081038344022e-05, 0.1: 3.503833535678513e-05, 0.2: 5.838048329875889e-05, 0.3: 6.944455949908779e-05}.
Confirmation target: 0.640; reached: False.

Weight selected on folds0-1; folds2-4 untouched confirmation.

These are local held-series OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.

Report replay: the frozen report called Model26's depth-mix loader with a
missing checkpoint directory. `report_fix.py` supplies the frozen Model26
directory and protocol fingerprint at runtime; it does not change any fit,
prediction, or predeclared comparison.
