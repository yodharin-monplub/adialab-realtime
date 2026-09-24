# Model 7: TSAUC-aligned ranking

Model 7 tests whether direct within-time ranking, more complete-series training
data, a low-capacity ensemble, and causal absorbing-state smoothing improve on
Model 6. It uses only Model 6's 6,666 development series. The 1,334-series
Model 6 holdout has already been revealed and is prohibited from every Model 7
fit, selection, and score; Model 7 intentionally has no holdout command.

## Protocol

Three deterministic five-fold assignments are created inside the allowed
development population. For each scored fold, the next fold is reserved for
early stopping and the remaining three fit the model. Complete series—not
individual rows—are sampled up to each row cap. A weighted Fisher ranking is
rebuilt from the fitting role and selects 256 candidate-family columns.

All three models are predeclared and receive all 15 held-fold fits:

- XGBoost `rank:pairwise`, 800,000-row complete-series cap;
- the same ranker with a 1,200,000-row cap;
- LightGBM GOSS binary baseline with a 1,200,000-row cap.

For XGBoost, a query is one exact `time_online` stratum. Queries with only one
class are removed from fitting because they have zero TSAUC pair weight. Query
weights are `n_positive * n_negative`, and early stopping uses this
repository's exact TSAUC implementation rather than XGBoost's different
built-in grouped AUC.

The ensemble is nested. For every held fold, only its other four folds choose
the two components and one nonnegative logit-blend weight. Selection maximizes
`mean TSAUC - 0.5 * SE`. A causal Gaussian-emission absorbing-state filter is
tested the same way, with the raw blend retained as an allowed endpoint.
Uncertainty is reported across repeat means and with a 200-replicate
complete-series bootstrap of predictions averaged across repeats.

## Commands

Run from the repository root:

```bash
python3 models/model7/run_model7.py status
/home/hp/jupyter-lab-venv/bin/python models/model7/run_model7.py screen
/home/hp/jupyter-lab-venv/bin/python models/model7/run_model7.py repeated-cv
/home/hp/jupyter-lab-venv/bin/python models/model7/run_model7.py ensemble
```

`run-all` executes or resumes every missing stage. Each fit writes its model,
selected columns, prediction, and hash-checked JSON record under the
configuration-derived `models/model7/runs/` directory. The short screen is
informational only and cannot eliminate a model from final CV.

## Interpretation

Repeated folds reuse the same allowed series under different assignments, so
the repeat-level SE and clustered bootstrap are more meaningful than treating
all 15 fold scores as independent. These are development estimates, not a new
external holdout. Do not inspect or score Model 6 holdout artifacts to select a
Model 7 architecture.

## Completed result

The fixed protocol completed under run ID `rankcv_51e67bff00`:

| Architecture | Mean repeat TSAUC | Repeat SE |
| --- | ---: | ---: |
| XGBoost pairwise, 800k rows | 0.601051 | 0.001490 |
| XGBoost pairwise, 1.2M rows | 0.606227 | 0.001402 |
| LightGBM GOSS, 1.2M rows | 0.606247 | 0.002175 |
| Nested two-model blend | **0.610682** | **0.002051** |

The blend's repeat scores were `0.610196`, `0.607397`, and `0.614452`.
Predictions averaged across repeats scored 0.617209; a 200-replicate
complete-series bootstrap gave SE 0.005394 and percentile interval
`[0.607136, 0.626831]`. The nested selector retained the raw blend in all 15
held folds, so the absorbing-state filter did not improve this experiment.
The 0.64 target was not reached. In particular, ordinary row ROC AUC was
0.644632 while TSAUC was materially lower, demonstrating that ordinary AUC is
not a valid model-selection substitute here.

The detailed immutable reports are
`runs/rankcv_51e67bff00/repeated_cv/summary.json` and
`runs/rankcv_51e67bff00/ensemble/summary.json`. These results do not justify a
submission bundle; the prior Model 6 candidate architecture remains stronger
on its nested development estimate.

## Complete source copy

[`source/`](source/) contains a hash-manifested mini-repository with Model 7,
its upstream model packages, shared pipelines, imported detectors, and tests.
Generated experiment output and training data are excluded.
