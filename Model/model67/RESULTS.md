# Model67: fixed 50/25/25 held-series ensemble

| Arm | Confirmation TSAUC | Full OOF TSAUC | Gain vs Model62 | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model39 | 0.638793 | 0.634268 | — | — | — | — |
| model62 | 0.639785 | 0.637534 | baseline | baseline | baseline | baseline |
| 50% m62 + 25% m63 + 25% m64 | 0.640196 | 0.637609 | +0.000410 | 3/3 | +0.000015 | False |

Confirmation split-half gains vs Model62: [0.0004636330287355328, 0.00037780325055003505].
Exploratory held-series OOF; all three parent model choices reused development folds.

No packaging, upload, or cloud run performed.
