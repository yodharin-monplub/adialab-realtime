# Model 6: TSROC selection and multi-model ensemble

Model 6 starts from Model 5's train-only detector screen. It selects a bank
only when its five-fold lower confidence score satisfies
`mean TSROC - 0.5 * SE > 0.55`. No competition test data, protected labels, or
pretrained external model is used.

## Current 10,000-series refresh

The 2026-08-26 refresh uses Model 5 outer seed `20260826`, recorded CV seed
`20260839`, and Model 6 search seed `20260861`. Its deterministic run paths are:

- Model 5 detector screen: `models/model5/artifacts`
- Model 6 search: `models/model6/runs/tsroc0p64_909b8f5c6c`

Scores are written here only after the development protocol finishes and the
new frozen holdout is evaluated once. Results from the former 8,000-series
dataset are not comparable and must not be reused as checkpoints.

The post-processing search now compares the raw prediction, the original
additive logit-space decayed maximum, and the requested probability recurrence:

```text
state[t] = max(raw_probability[t], decay_factor * state[t - 1])
```

The multiplicative factor grid is `1.0, 0.9999, 0.999, 0.995, 0.99, 0.98,
0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50`. Factor `1.0` is a strict running
maximum. The grid is dense near one because a small difference compounds over
hundreds of yields. For each scored development fold, the transform is chosen
using only the other four folds; the deployment factor is then selected from
all development folds before the outer labels are opened.

## Leakage controls

The 10,000 training series are assigned once to six complete-series folds.
Outer fold 0 is sealed; the other five are development folds. For a scored
development fold, the next fold is reserved for early stopping and the
remaining folds fit the learner. Feature rankings are rebuilt per score fold,
so gain, Fisher separation, and correlation pruning cannot see that fold.

The search is deliberately staged:

1. Test gain, split, consensus, correlation-pruned gain, and weighted Fisher
   rankings at 32, 64, 128, 256, 512, and all features.
2. On promoted subsets, compare LightGBM GBDT/DART/GOSS/RF, XGBoost,
   CatBoost, RandomForest, ExtraTrees, and ridge-, lasso-, and elastic-net-like
   SGD logistic models.
3. Tune promoted learners, then produce complete five-fold OOF predictions for
   three finalists per family.
4. Cross-fit detector-level stackers, build larger core/candidate ensembles,
   and greedily blend them. Causal persistence is selected without each held
   fold.
5. Freeze a hashed deployment before scoring the outer holdout once. The
   reported SE is a complete-series cluster bootstrap, never a row-level SE.

## Commands

Run from the repository root with the Jupyter environment:

```bash
.venv/bin/python models/model6/run_model6.py prepare-source
.venv/bin/python models/model6/run_model6.py status
.venv/bin/python models/model6/run_model6.py run-development
.venv/bin/python models/model6/run_model6.py run-all
```

For the full 10,000-series rebuild as a durable background job, run
`bash models/model6/run_refresh_10000.sh`. It serializes the cache, Model 5
source-screen, and Model 6 stages under a file lock and appends progress to
`models/model6/refresh_10000.log`.

Install `requirements.txt` into the repository `.venv` first. The wrapper
prefers `.venv/bin/python`, accepts `MODEL6_PYTHON_BIN` as an explicit
override, and checks the LightGBM, XGBoost, CatBoost, and scikit-learn imports
before starting expensive work.

`prepare-source` builds/validates the all-detector caches, runs Model 5's
fresh detector screen, and creates the fold-excluded selected-family baseline
models required for Model 6 feature rankings, without opening the outer
holdout. Both
`run-development` and `run-all` invoke this preparation automatically.
`run-development` never opens the outer holdout. `run-all` resumes all missing
checkpoints, freezes the winner, builds holdout predictions without labels,
then opens the holdout once. Outputs are routed to configuration-hashed
directories; rerunning is safe.

## Hardware note

The code probes the NVIDIA driver, not merely installed CUDA wheels. On this
machine `nvidia-smi` is absent, LightGBM has no usable OpenCL/CUDA device,
CatBoost reports an incompatible CUDA driver, and XGBoost falls back to CPU.
The manifest therefore records `actual_device: cpu` and uses up to 12 threads.
If a working NVIDIA driver becomes visible, `requested_device="auto"` enables
supported GPU backends on a new configuration-hashed run.

## Configuration

Edit [config.py](config.py), restart the Python/Jupyter process, and rerun.
Changing any model-affecting value creates a different run directory; old
checkpoints are not mixed. Increase promotion widths and row caps for a wider,
slower search. Do not alter or inspect holdout artifacts after `freeze` to tune
another candidate.

## Complete source copy

[`source/`](source/) contains a hash-manifested mini-repository with Model 6,
its upstream model packages, shared pipelines, imported detectors, and tests.
Generated fits, caches, predictions, logs, and training data are excluded.
