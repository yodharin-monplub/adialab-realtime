# Model 12 results

| Candidate | Mean fold TSAUC | Fold SE |
|---|---:|---:|
| model9_pairwise_stack + 10% Model 12 | 0.628380 | 0.007581 |
| model9_pairwise_stack + 25% Model 12 | 0.628302 | 0.007546 |
| model6_incumbent_postprocess + 10% Model 12 | 0.625481 | 0.008393 |
| model9_pairwise_stack + 50% Model 12 | 0.625026 | 0.007445 |
| model6_incumbent_postprocess + 25% Model 12 | 0.624043 | 0.008224 |
| model6_incumbent_postprocess + 50% Model 12 | 0.617535 | 0.008088 |
| Model 12 standalone | 0.582356 | 0.006683 |

All five complete-series development folds finished. No outer-holdout
or competition test labels were loaded. These are local development
diagnostics; stacking/blending inherits upstream adaptive selection.

No deployment was promoted. See `../NEXT_RESULTS.md` for comparisons
and the best candidate's conditional whole-series bootstrap.
