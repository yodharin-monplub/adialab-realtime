# Model 8: causal detector expansion

Model 8 evaluates six new candidate detector families derived from the
remaining ideas in `models/past_solutions/summary.md`. It uses only Model 6's
6,666-series development population. The previously revealed 1,334-series
outer holdout and every Model 6 holdout prediction are prohibited.

## New banks

| Bank | Features | Evidence |
| --- | ---: | --- |
| `d61` | 28 | Wilcoxon/rank shift, quantile-grid Anderson–Darling, energy and tail balance |
| `d62` | 36 | Fixed-age Sup-F-like step, scale and piecewise-regression scans |
| `d63` | 30 | Symbolic sample entropy, BDS gap, transition MI and ordinal entropy |
| `d64` | 49 | Robust time trends, Fisher ACF/PACF changes and magnitude trends |
| `d65` | 36 | Label-free Fisher, Cauchy and signed-Stouffer window aggregation |
| `d66` | 60 | IQR-winsorized changes and causal stationary Symlet-2 energy |

Each detector freezes its reference and calibration from clean history, uses
only the online prefix ending at the current row, and has independently tested
offline and streaming paths. The 239-column cache is built only from `train/`.

## Validation protocol

Model 8 retains Model 6's five complete-series development folds. For held
fold `f`, fold `(f + 1) mod 5` supplies early stopping and the other three
folds supply training rows. Row caps sample complete series without labels.
Every bank receives five LightGBM fits.

The fixed Model 6 candidate `nested_oof` prediction is the baseline. It is
cross-fitted and contains no outer-holdout rows. A detector passes only when
both its standalone `mean TSAUC - 0.5 × SE` exceeds 0.5 and its nested
baseline-blend improvement clears zero. Fold-specific bank selection repeats
that rule using only the other four folds.

The predeclared primary architecture trains one LightGBM model on all six
banks for each fold, then chooses its baseline-blend weight without that fold.
Secondary diagnostics are a fold-nested selected-bank model and a nested
greedy blend of detector OOF predictions. The primary receives a 200-replicate
complete-series bootstrap.

## Commands

Run from the repository root with the project environment:

```bash
MPLCONFIGDIR=/tmp/matplotlib-model8 /home/hp/jupyter-lab-venv/bin/python models/model8/run_model8.py status
MPLCONFIGDIR=/tmp/matplotlib-model8 /home/hp/jupyter-lab-venv/bin/python models/model8/run_model8.py build-cache
MPLCONFIGDIR=/tmp/matplotlib-model8 /home/hp/jupyter-lab-venv/bin/python models/model8/run_model8.py screen
MPLCONFIGDIR=/tmp/matplotlib-model8 /home/hp/jupyter-lab-venv/bin/python models/model8/run_model8.py final
MPLCONFIGDIR=/tmp/matplotlib-model8 /home/hp/jupyter-lab-venv/bin/python models/model8/run_model8.py freeze-outer
MPLCONFIGDIR=/tmp/matplotlib-model8 /home/hp/jupyter-lab-venv/bin/python models/model8/run_model8.py predict-outer
MPLCONFIGDIR=/tmp/matplotlib-model8 /home/hp/jupyter-lab-venv/bin/python models/model8/run_model8.py evaluate-outer
```

`run-all` resumes every missing stage. Fold predictions, models, feature
contracts and JSON reports are checkpointed below the configuration-derived
`runs/` directory. A changed configuration receives a different run ID.
`run-all` deliberately remains development-only: opening the outer split
requires the explicit three-stage freeze/predict/evaluate sequence above.

## Results

The fixed experiment completed on 2026-08-26. Individual detector screening
gave:

| Bank | Mean TSAUC | Fold SE | Robust blend gain | Selected |
| --- | ---: | ---: | ---: | :---: |
| `d61` | 0.534173 | 0.004311 | +0.000020 | yes |
| `d62` | 0.507541 | 0.005993 | 0.000000 | no |
| `d63` | 0.510046 | 0.004131 | 0.000000 | no |
| `d64` | 0.512243 | 0.003187 | 0.000000 | no |
| `d65` | 0.538598 | 0.005618 | -0.000351 | no |
| `d66` | 0.544832 | 0.008036 | -0.000040 | no |

The model-level results were:

| Architecture | Mean TSAUC | Fold SE | Pooled TSAUC |
| --- | ---: | ---: | ---: |
| Model 6 candidate baseline | **0.622097** | 0.004856 | 0.620585 |
| All six, raw | 0.562392 | 0.008235 | 0.557973 |
| All six + nested baseline blend (primary) | 0.621595 | 0.005260 | 0.620094 |
| Fold-nested selected banks, raw | 0.561576 | 0.008525 | 0.561470 |
| Nested greedy bank blend | 0.621823 | 0.005067 | **0.620649** |

The primary's paired development-fold change from baseline was -0.000502 with SE
0.000490. Its complete-series bootstrap SE was 0.005712 and its percentile
95% interval was [0.608887, 0.630021]. The 0.64 target was not reached. The
new features are useful as documented research, but this experiment provides
no evidence to replace the current Model 6 candidate baseline or create a
submission bundle. Full machine-readable results are in
`runs/newdet_514d252fb4/final/summary.json`.

### Explicit outer-holdout rerun

At the user's request, the existing 1,334-series outer split was subsequently
opened under a separate frozen contract. Before prediction or scoring, the
development-only robust rule fixed the deployment weight at 5% Model 8 and
95% Model 6 candidate. Five Model 8 voters each trained on four development
folds and used the fifth only for early stopping; their logits were averaged.

| Frozen outer prediction | TSAUC |
| --- | ---: |
| Model 6 candidate baseline | 0.604658 |
| Raw all-six Model 8 diagnostic | 0.566204 |
| **Frozen 95/5 primary** | **0.605003** |

The primary-minus-baseline change was +0.000345. The primary complete-series
bootstrap SE was 0.013386 and its 95% interval was [0.579009, 0.633111]. It did
not reach 0.64. The immutable contract hash is
`b60f9f456e3941fc5d8740a832c31b4993171338e5f534fb86437edc5f37ec78`; full
results are in `runs/newdet_514d252fb4/outer_holdout/results.json`.

This is procedurally honest for Model 8—no fit, feature, or weight was changed
after its outer score—but it is not a pristine estimate: Model 6 had already
opened the same repository holdout, and the broader research trajectory knew
that result. It should not be treated as an unbiased competition-score
forecast.

## Complete source copy

[`source/`](source/) contains a hash-manifested mini-repository with Model 8,
its upstream model packages, shared pipelines, every imported detector, and
tests. Generated experiment output and training data are excluded.
