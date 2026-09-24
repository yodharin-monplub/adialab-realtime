# Model 12

ExtraTrees diversity model using Model 11's 154 strictly causal features.

The shared implementation is in `models/model11/`. Features compare six
trailing windows (8, 16, 32, 64, 128, 256) with the guaranteed-clean history.
They measure changes in means, scales, tail rates and lag dependence on levels,
first differences, AR(1) residuals and historical empirical percentile ranks.
Historical length and elapsed online time are available; final online length
is never a feature. Streaming state resets for every series.

Run from the project root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python3 models/run_next_models.py run --models 12
python3 tests/test_compact_models.py
```

The immutable recipe lives in `models/model11/config.py`. Training uses CPU,
fixed boosting/tree counts, deterministic row sampling after complete-series
fold assignment, and all four other development folds. No early stopping or
feature selection reads the held fold. Feature construction is unsupervised.
The original 1,667 outer-holdout series remain excluded from labels and fits.

Dependencies: official `train/X_train.parquet` and
`train/y_train_index.parquet`, Model 5 split assignments, restored unlabeled
row metadata, and saved Model 6 / Model 9 OOF vectors for comparisons.
`python3 models/download_training.py` retrieves only the training files.
The API credential is read at runtime and is not embedded in model code.

The shared feature cache is under `models/artifacts/compact_multiscale/`.
Runs are identified by data, feature source, pipeline, recipe and split hashes;
completed folds are reused. Each run retains five fitted models, OOF predictions,
fold reports, a protocol and a summary. Raw results use repeatedly reused
development data. Predeclared 0.10/0.25/0.50 logit blends with Models 6 and 9
inherit their upstream validation limitations and are exploratory diagnostics.
No automatic promotion or competition submission occurs.

`main.py` provides Crunch train/infer entry points, backed by the shared
serving code. Inference emits a readiness sentinel and consumes exactly one
online observation before yielding each score. Those entry points require the
shared packages when bundled. CV artifacts are research checkpoints, not an
automatically promoted submission.
