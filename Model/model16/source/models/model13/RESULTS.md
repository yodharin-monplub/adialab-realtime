# Model 13 results

| Candidate | Mean fold TSAUC | Fold SE |
|---|---:|---:|
| model9_pairwise_stack + 10% Model 13 | 0.628238 | 0.007565 |
| model9_pairwise_stack + 25% Model 13 | 0.626873 | 0.007382 |
| model6_incumbent_postprocess + 10% Model 13 | 0.625000 | 0.008346 |
| model6_incumbent_postprocess + 25% Model 13 | 0.620829 | 0.008032 |
| model9_pairwise_stack + 50% Model 13 | 0.618049 | 0.006769 |
| model6_incumbent_postprocess + 50% Model 13 | 0.607453 | 0.007761 |
| Model 13 standalone | 0.566606 | 0.006505 |

All five complete-series development folds finished. No outer-holdout
or competition test labels were loaded. These are local development
diagnostics; stacking/blending inherits upstream adaptive selection.

No deployment was promoted. See `../NEXT_RESULTS.md` for comparisons
and the best candidate's conditional whole-series bootstrap.
