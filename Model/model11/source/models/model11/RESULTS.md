# Model 11 results

| Candidate | Mean fold TSAUC | Fold SE |
|---|---:|---:|
| model9_pairwise_stack + 10% Model 11 | 0.628595 | 0.007474 |
| model9_pairwise_stack + 25% Model 11 | 0.627866 | 0.007176 |
| model6_incumbent_postprocess + 10% Model 11 | 0.625726 | 0.008179 |
| model6_incumbent_postprocess + 25% Model 11 | 0.622609 | 0.007714 |
| model9_pairwise_stack + 50% Model 11 | 0.620325 | 0.006542 |
| model6_incumbent_postprocess + 50% Model 11 | 0.610359 | 0.007339 |
| Model 11 standalone | 0.570593 | 0.005488 |

All five complete-series development folds finished. No outer-holdout
or competition test labels were loaded. These are local development
diagnostics; stacking/blending inherits upstream adaptive selection.

No deployment was promoted. See `../NEXT_RESULTS.md` for comparisons
and the best candidate's conditional whole-series bootstrap.
