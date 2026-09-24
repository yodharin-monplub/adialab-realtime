# Model30: lower-shift matched augmentation

| Arm | Component OOF | Full OOF | Gain | Wins | Lower 95% | Pass |
|---|---:|---:|---:|---:|---:|---|
| model26 | 0.618445 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline |
| matched15_50 | 0.620809 | 0.631509 | -0.000898 | 0/5 | -0.001498 | False |

Target held-series OOF TSAUC: 0.640; reached: False.

Exploratory matched-augmentation backbone selected after Model27 showed fold-specific failure from appended history context. Model30 reuses Model27's frozen positive/hard-negative composites but removes all context columns, lowers synthetic mass from 25% to 15%, uses depth 3, and fixes a 50/50 blend with Model26. Every prediction remains held-series OOF.

These are local OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
