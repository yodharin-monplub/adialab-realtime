# Model 10 Results

Model 10 evaluated 128 endpoints: the unchanged incumbent plus EMA
smoothing/trend, finite-window maximum blending, soft decayed peaks, and
prior-only normalized innovation. The run used 4,195,720 development OOF rows
from 8,333 complete series. Sealed-series labels were neither materialized nor
used, and no test labels were opened.

| Frozen source | Raw TSAUC ± fold SE | Nested gain ± paired SE | Wins | Decision |
|---|---:|---:|---:|---|
| Model 6 incumbent | 0.625675 ± 0.008440 | -0.000299 ± 0.000600 | 1/5 | reject |
| Model 9 pairwise stack | 0.628020 ± 0.007603 | +0.000289 ± 0.000963 | 2/5 | reject |

The nested gain is computed on untouched held cells. Cell TSAUC changes are
weighted by their official positive-negative pair counts within each original
fold, then uncertainty is calculated over the five original folds. Both
sources fail the predeclared robust-gain and four-of-five-wins gates, so no
post-processing transform is promoted and no submission was changed.

The strongest same-data diagnostics were:

- Model 6: logit EMA smoothing, half-life 64 and weight 0.10, gained only
  `+0.000102` and remained below the minimum-effect gate.
- Model 9: a 128-row rolling maximum blended at weight 0.10 gained
  `+0.000741 ± 0.000426` across the same five folds, with 4/5 wins. This is
  post-hoc and Model 9's upstream stack is exploratory, so it is a candidate
  for future fresh validation rather than deployment evidence.

Machine-readable results and resumable grid scores are under
`runs/postprocess_e4ef830f0e/`. A global monotone calibration was omitted
because it cannot change TSAUC's within-time rankings.
