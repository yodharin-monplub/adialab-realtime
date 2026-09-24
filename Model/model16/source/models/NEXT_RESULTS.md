# Models 11–16 results

CPU development experiments on 8,333 complete series / 4,195,720 online rows.

| Candidate | Mean fold TSAUC | Fold SE |
|---|---:|---:|
| Model 6 incumbent | 0.625675 | 0.008440 |
| Model 9 exploratory stack | 0.628020 | 0.007603 |
| Model 10 historical rejected diagnostic | 0.628761 | 0.007858 |
| Model 16 persistence + diversity | 0.629300 | 0.007709 |
| model9_pairwise_stack + 10% Model 11 | 0.628595 | 0.007474 |
| model9_pairwise_stack + 10% Model 12 | 0.628380 | 0.007581 |
| model9_pairwise_stack + 25% Model 12 | 0.628302 | 0.007546 |
| model9_pairwise_stack + 10% Model 13 | 0.628238 | 0.007565 |
| Model 14 time-varying stack | 0.628160 | 0.007483 |
| model9_pairwise_stack + 25% Model 11 | 0.627866 | 0.007176 |
| model9_pairwise_stack + 25% Model 13 | 0.626873 | 0.007382 |
| model6_incumbent_postprocess + 10% Model 11 | 0.625726 | 0.008179 |
| model6_incumbent_postprocess + 10% Model 12 | 0.625481 | 0.008393 |
| model9_pairwise_stack + 50% Model 12 | 0.625026 | 0.007445 |
| model6_incumbent_postprocess + 10% Model 13 | 0.625000 | 0.008346 |
| model6_incumbent_postprocess + 25% Model 12 | 0.624043 | 0.008224 |
| Model 15 residual correction | 0.623299 | 0.006920 |
| model6_incumbent_postprocess + 25% Model 11 | 0.622609 | 0.007714 |
| model6_incumbent_postprocess + 25% Model 13 | 0.620829 | 0.008032 |
| model9_pairwise_stack + 50% Model 11 | 0.620325 | 0.006542 |
| model9_pairwise_stack + 50% Model 13 | 0.618049 | 0.006769 |
| model6_incumbent_postprocess + 50% Model 12 | 0.617535 | 0.008088 |
| model6_incumbent_postprocess + 50% Model 11 | 0.610359 | 0.007339 |
| model6_incumbent_postprocess + 50% Model 13 | 0.607453 | 0.007761 |
| Model 12 standalone | 0.582356 | 0.006683 |
| Model 11 standalone | 0.570593 | 0.005488 |
| Model 13 standalone | 0.566606 | 0.006505 |

Best observed candidate: Model 16 persistence + diversity.

Paired bootstrap resamples complete series within each original fold:

| Reference | Gain | Bootstrap SE | Percentile 95% interval |
|---|---:|---:|---:|
| model6_incumbent_postprocess | +0.003625 | 0.002124 | [-0.000644, +0.007462] |
| model9_pairwise_stack | +0.001280 | 0.000588 | [+0.000259, +0.002461] |
| historical_model10_rejected_diagnostic | +0.000539 | 0.000483 | [-0.000377, +0.001658] |

Paired complete-series resampling within original folds, conditional on saved predictions; excludes upstream refitting and adaptive model-selection uncertainty.

Keep Model 6 deployment. Candidate evidence remains exploratory; no submission was made.

The outer holdout and competition test labels were not loaded. These are local
development scores, not leaderboard scores or evidence of a prize-winning rank.

Reproduce: `python3 models/run_next_models.py run`, then
`python3 models/model14/run_model14.py`, then
`python3 models/model15/run_model15.py`, then
`python3 models/model16/run_model16.py`, then
`python3 models/report_next_models.py --bootstrap 200`.
