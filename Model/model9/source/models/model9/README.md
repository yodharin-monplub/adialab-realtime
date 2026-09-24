# Model 9: full-row scaling and exploratory stacking

Model 9 tests whether Model 6 was limited by its row cap rather than by its
detectors. It reuses train-derived feature matrices, fold-excluded feature
rankings, and cross-fitted Model 6 predictions. The competition test set is not
used. The previously revealed outer holdout is sealed: this package contains no
command that predicts or scores it.

This is not a fresh leakage-free validation study. Model 5 selected the 31
upstream detector banks using these same five development folds, and Models 6–9
have repeatedly used the development results. Raw scores for a fixed Model 9
architecture are therefore reused-development, conditional evidence. Use them
to compare the predeclared row scales, not as an unbiased competition estimate.

## Validation contract

The 8,333-series development population keeps Model 6's five fixed folds. For
score fold `f`, fold `(f + 1) mod 5` is the early-stopping fold and the other
three folds are eligible for fitting. Complete series are assigned to disjoint
fit, early-stop, and score roles. Within a fit or stop role, row caps use one
deterministic, label-independent row permutation across all eligible series;
the 400,000-row sample is therefore an exact subset of the 1,200,000-row sample.
The score fold is never row-subsampled. Fit and stop caps sample rows, not whole
series, after the series-disjoint role assignment.

Four predeclared LightGBM architectures are compared at 400,000 and 1,200,000
fit rows:

| Architecture | Feature contract |
| --- | --- |
| `candidate_all_gbdt` | all 1,415 candidate columns |
| `candidate_split512_gbdt` | 512 fold-excluded split-ranked candidate columns |
| `core_fisher512_gbdt` | 512 fold-excluded Fisher-ranked core columns |
| `joint_fisher768_gbdt` | 256 core plus 512 candidate Fisher-ranked columns |

Selection uses mean fold TSAUC minus `0.5 × fold SE`. The code excludes a held
fold's score scalar when choosing architectures, transforms, and blend weights.
That does not make the higher-level result fully nested: an OOF voter for a
different fold may have trained or early-stopped on the held fold. Confirmatory
selection would require refitting every base voter inside each outer-held split.

Consequently, causal postprocessing, architecture selection, blends, promotion,
the pairwise stack, and the shallow LightGBM meta gate are all exploratory. The
pairwise and meta inputs also reuse primitive models selected on these folds.
Their fold SE and complete-series bootstrap describe the saved OOF predictions;
they do not include base-refit or adaptive-selection uncertainty. No Model 9
result is a new outer-holdout or leaderboard score.

## Run or resume

Run commands from the repository root:

```bash
python3 models/model9/run_model9.py status
python3 models/model9/run_model9.py pairwise-stack
python3 models/model9/run_model9.py scale-cv
python3 models/model9/run_model9.py promote
python3 models/model9/run_model9.py meta-stack
python3 models/model9/run_model9.py run-all
```

Every completed fold writes its model, feature contract, prediction, and JSON
record below the configuration-derived `models/model9/runs/fulldata_*/`
directory. Repeating a command validates and reuses matching checkpoints, so an
interrupted run can safely resume. A protocol-changing configuration produces a
new run ID instead of mixing artifacts. `model9.ipynb` exposes the same stages
as separate, commented cells.

## Resource expectations

The scaling pipeline is a CPU LightGBM experiment; CUDA is not required.
There are 40 scale fits, followed by 5 fits per promoted architecture and five
small meta fits. On the repository machine (4 physical CPU cores, about 30 GiB
RAM), plan roughly 4–7 hours for scale CV and another 5–8 hours for promotion;
actual time depends strongly on how many architectures are promoted. Pairwise
and meta stages are comparatively small.

The 1,415-column 2,000,000-row fit is the peak-memory case: its raw float matrix
is about 10.5 GiB before the stop set and LightGBM overhead, so close other
memory-heavy programs and monitor RAM. Check `df -h .` before starting; keep at
least 10 GiB free for checkpoints and temporary files. Do not delete or alter
Model 6 artifacts while Model 9 is running because its feature matrices,
rankings, and baseline OOF prediction are read-only dependencies.

## Reading results

Compact output reports `fold_mean`, `fold_standard_error`, `robust_score`, and
`pooled_tsroc`. First compare each architecture's `raw` score at 400,000 versus
1,200,000 rows. Those are the least adaptive Model 9 results, but remain
conditional on upstream Model 5 bank selection. Treat every `postprocessed`,
`nested_*`, promoted, pairwise, meta, and bootstrap result as exploratory. A
small gain with overlapping uncertainty is not evidence of improvement.

The recorded deployment choices are experiment recipes, not a submission
bundle. A winner still needs an inference implementation and evaluation on a
genuinely untouched external split before replacing Model 6.

## Complete source copy

[`source/`](source/) contains a hash-manifested mini-repository with Model 9,
its upstream model packages, shared pipelines, imported detectors, and tests.
Generated experiment output and training data are excluded.
