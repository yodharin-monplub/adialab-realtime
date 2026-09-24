# Model39: smoothed anti-residual composition

| Arm | Held-series OOF | Gain vs M26 | Wins vs M26 | Lower vs M26 | Pass |
|---|---:|---:|---:|---:|---|
| model26 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline |
| model35 | 0.634058 | +0.001651 | 3/5 | diagnostic | exploratory |
| model39 | 0.634268 | +0.001861 | 3/5 | -0.000445 | False |

Model39 gain vs Model35: +0.000210; wins 5/5; bootstrap lower 95% +0.000112.
Split-half gains vs Model26: [2.8828590398322085e-05, 0.0034770695948891372].
Target: 0.640; reached: False.

Exploratory Model35 margin with independently selected and confirmed Model38 smoother.

These are local development OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
