# Model40: two-pass smoothed anti-residual

| Arm | Held-series OOF | Gain vs M26 | Wins vs M26 | Lower vs M26 | Pass |
|---|---:|---:|---:|---:|---|
| model26 | 0.632407 | +0.000000 | 0/5 | +0.000000 | baseline |
| model39 | 0.634268 | +0.001861 | 3/5 | diagnostic | exploratory |
| model40 | 0.634335 | +0.001928 | 3/5 | -0.000332 | False |

Gain vs Model39: +0.000067; wins 5/5; bootstrap lower 95% -0.000017.
Split-half gains vs Model26: [3.188218640604035e-05, 0.0035741012082033397].
Target: 0.640; reached: False.

Selection status: adaptively selected after inspecting one, two, and three passes on reused development OOF.

These are local development OOF scores, not cloud scores.

No packaging, upload, or cloud run performed.
