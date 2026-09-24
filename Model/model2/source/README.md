# Structural Break: Real-Time solution

This repository contains a leakage-safe, reproducible model for CrunchDAO's
Structural Break: Real-Time competition.  It uses only the files under
`train/` for fitting and model selection.  The password-protected local test
archive is intentionally excluded from every training, feature-selection,
and validation path.

The selected deliverable is a 50/50 logit ensemble of two independently
trained LightGBM models over causal, per-series change-detection features. It
emits one probability immediately after each new online observation. The same
feature mathematics has independent vectorized (training) and incremental
(submission) implementations, with regression tests requiring numerical
equality.

## What is being optimized

The target is not ordinary row-level ROC AUC.  The official scorer computes
ROC AUC across all series still present at online step `t`, then averages over
time with the number of positive-negative pairs as weights:

```text
TS-AUC = sum_t [n_pos(t) * n_neg(t) * AUC(t)]
         -------------------------------------
              sum_t n_pos(t) * n_neg(t)
```

Consequences for the implementation:

- no pair of observations at different online times is compared;
- a pure function of elapsed time earns exactly 0.5;
- train/validation splitting must happen by whole time series, not by rows;
- row weights should respect the pair geometry at each time;
- validation must score every held-out online row, even though only a
  deterministic subset is needed to fit LightGBM.

[`tsauc.py`](tsauc.py) mirrors the official scorer and includes the exact
single-class-stratum behavior.  The upstream implementation is available in
[CrunchDAO's public scoring source](https://github.com/crunchdao/competitions/blob/master/competitions/structural-break-real-time/scoring/scoring.py).

## Audited training data

The loader validates the period ordering and verifies that labels generated
from `tau_index` exactly match `y_train.parquet`.

| Quantity | Value |
|---|---:|
| Labeled series | 8,000 |
| Online decisions | 4,019,784 |
| Series with a break | 3,974 |
| Series without a break | 4,026 |
| Historical length | 1,000–5,000 |
| Online length | 10–999 |
| Median online length | 501 |
| Break location (when present) | 0–984 |

Period 1 is the complete clean historical reference.  Period 2 is the online
segment.  At online index `t`, the label is `1` exactly when `tau_index >= 0`
and `t >= tau_index`.

## Feature model

The core contract has 1,168 float features.  All calibration quantities are
estimated from the guaranteed-clean history and then frozen.  Online states
are updated with only the current observation.

| Bank | Columns | Purpose |
|---|---:|---|
| `d4` | 306 | Location, scale, robust shape, tails, quantile crossings, serial dependence, AR(5) residuals, and lagged difference energy, each with prefix/CUSUM/multi-window scans |
| `d15` | 322 | Autocorrelation-adjusted EWMA evidence at seven response speeds |
| `d22` | 230 | Empirical historical-null calibration of suffix statistics |
| `d26` | 184 | Empirical historical-null calibration of selected EWMA statistics |
| `d27` | 70 | Empirical suffix calibration for extreme-tail and difference-energy specialists |
| `d30` | 56 | Empirical EWMA calibration for the same specialist channels |

The empirical-null transforms matter for a cross-sectional metric: a raw
CUSUM of 3 must mean comparable surprise on a quiet Gaussian series and a
heavy-tailed autocorrelated series.  Each series' clean history supplies that
normalization.

The complementary contract has 1,490 columns across `d32`, `d33`, `d35`,
`d37`, `d40`, `d45`, `d51`, `d52`, `d53`, `d55`, `d57`, and `d58`. It adds
Haar/wavelet energy, convolutional rank patterns, GARCH surprise, categorical
run shapes, calibrated sequential likelihoods, nonlinear forecast surprise,
and distribution/dependence alternatives. It is deliberately trained as a
separate booster: a single 2,658-column learner scored worse than the
two-family logit ensemble.

A third 1,322-column specialist cache was also built and validated. Its
distribution histograms, Fourier ECDF modes, long-lag energy, omnibus,
Dirichlet-multinomial, Haar, AR-score, copula, and local-CUSUM banks are kept as
a reproducible ablation, but excluded from deployment because they did not
improve the five-fold mean and substantially increased streaming cost.

`CoreStreamDetector` shares compatible detector state between transforms. This
avoids recomputing common innovations and preserves exact feature order.
Non-finite numerical corner cases (for example a constant history) are mapped
deterministically and tested.

## Validation design

Five folds are created at the **series** level.  Stratification jointly
balances:

1. break versus no break;
2. online-length quintile;
3. relative break-location quintile for broken series.

Each fold contains 1,600 complete series and about 795 broken series.  A fold
model sees no historical or online value from its validation series.

LightGBM fitting retains a deterministic 20% of rows to control memory.
Sampling is uniform and label-independent.  Every retained positive at time
`t` gets weight `n_neg(t)` and every retained negative gets weight
`n_pos(t)`, where counts are computed from the complete training portion of
that fold.  Thus both classes contribute equally inside a time stratum and
the total stratum emphasis follows the official `n_pos * n_neg` weighting.

Evaluation predicts all 4.02 million rows out of fold.  The reported standard
error is `sample_std(fold TS-AUC) / sqrt(5)`; the 95% interval uses a Student-t
critical value with four degrees of freedom.  This measures series-split and
model-fit variability, not uncertainty from repeatedly peeking at a local
test set.

Because the label is absorbing after a break, the report also evaluates a
small predeclared grid of causal decayed-maximum score states.  Raw OOF is the
primary estimate; a persistence penalty is deployed only if its improvement
is stable across every held-out fold.  [`postprocess.py`](postprocess.py)
contains vectorized and one-scalar streaming implementations of the same
recurrence.

## Reproduce

The environment used here has Python 3, NumPy, pandas, SciPy, scikit-learn,
LightGBM, joblib, and PyArrow.  Exact package bounds are in
[`requirements.txt`](requirements.txt).

```bash
# 1. Run metric, fold, causality, ensemble, and feature-parity tests.
python3 -m tests.run_tests

# 2. Build/resume both selected float32 feature caches.
python3 train_cv.py build-cache
python3 train_cv.py build-cache \
  --cache-name candidate_features \
  --banks d32 d33 d35 d37 d40 d45 d51 d52 d53 d55 d57 d58 \
  --workers 2

# 3. Train five folds for both independent families.
python3 train_cv.py cross-validate \
  --model-directory artifacts/model_core
python3 train_cv.py cross-validate \
  --cache-name candidate_features \
  --model-directory artifacts/model_candidate

# 4. Select weights without using each fold's own labels.
python3 ensemble_cv.py \
  artifacts/model_core artifacts/model_candidate \
  --weights 0.5 0.5 \
  --output artifacts/ensemble_cv_results.json

# 5. Use all labeled series at each family's CV-median iteration count.
python3 train_cv.py fit-final \
  --model-directory artifacts/model_core
python3 train_cv.py fit-final \
  --cache-name candidate_features \
  --model-directory artifacts/model_candidate

# 6. Verify one-pass inference and build the small portable bundle.
python3 validate_submission.py --model-directory artifacts/model_ensemble
python3 build_submission_bundle.py --output submission_ready
```

The cache has a SHA-256 signature over its version, bank order, and every
feature name.  Extraction is resumable at series boundaries.  Model folders
also carry a configuration hash, so incompatible arguments cannot silently
reuse stale models or predictions.

To run one fold first as a smoke/performance test:

```bash
python3 train_cv.py cross-validate \
  --model-directory artifacts/model_core \
  --only-fold 0
```

Rerun the command without `--only-fold` to resume and finish the other folds.

## Results and uncertainty

The primary estimate is the nested blend: for each held fold, its core weight
was selected on the other four folds. All five selections were 0.475–0.500;
their median fixes deployment at exactly 0.500. The fold standard error is
`sample_std / sqrt(5)` and the interval uses Student-t with four degrees of
freedom.

| Model | Fold mean TS-AUC | SE | 95% t interval | Pooled OOF |
|---|---:|---:|---:|---:|
| Core (1,168 columns) | 0.617589 | 0.005429 | [0.602516, 0.632662] | 0.616349 |
| Candidate (1,490 columns) | 0.613011 | 0.006592 | [0.594709, 0.631312] | 0.611964 |
| Equal logit blend | 0.627676 | 0.005578 | [0.612188, 0.643164] | 0.626482 |
| **Nested weight-selection procedure** | **0.627612** | **0.005612** | **[0.612031, 0.643193]** | n/a |

The deployable equal-blend fold scores are `0.633421`, `0.616268`,
`0.645712`, `0.616134`, and `0.626847`. The nested estimate is the honest
headline because the weight applied to a held fold never used that fold's
labels. Machine-readable reports are in
`artifacts/ensemble_cv_results.json`, `artifacts/model_core/cv_results.json`,
and `artifacts/model_candidate/cv_results.json`.

### Rejected experiments

Every material alternative is retained in code/artifacts instead of being
silently omitted:

| Experiment | Result | Decision |
|---|---:|---|
| One joint 2,658-column learner, fold 0 | 0.626153 | Worse than separate blend, 0.633718 |
| Specialist alone, five-fold mean | 0.599128 | Weak standalone model |
| Core/candidate/specialist blend | 0.627640 ± 0.005849 | Worse mean and SE than two-family blend |
| Nested nonlinear stacker | 0.626661 ± 0.005387 | Worse than equal logit blend |
| Nested time-regime weights | 0.627325 ± 0.005551 | Worse; weights remained near 0.5 |
| 40% rather than 20% rows, fold 0 | core 0.621371; candidate 0.616796 | Both below 20% counterparts |
| 63-leaf core, fold 0 | 0.616376 | Below selected 31-leaf model, 0.622562 |
| Decayed-maximum persistence | inconsistent by fold | Disabled |

No training-only estimate establishes a 0.65 leaderboard result. The selected
model is the strongest design that survived the leakage-safe tests here; it
does not justify a promise of first place on unseen data.

## Competition submission

[`solution.py`](solution.py) exposes the required functions:

```python
train(datasets, model_directory_path)
infer(datasets, model_directory_path)
```

`infer` is a generator. It yields the required readiness sentinel once and
then exactly one finite probability for each online observation. It never
asks for the stream length, iterates a stream twice, carries state between
series, or touches the protected archive. The selected bundle loads one
all-data core booster and one all-data candidate booster, combines their
probabilities in log-odds space, and computes the union of detector states
once.

On the development machine, a 250-step series took 0.299 s for detector
initialization plus its first score and 4.02 ms per subsequent step. The two
final LightGBM files total about 1.21 MB. `validate_submission.py` checks
one-pass consumption, readiness, output count/range, and bit-for-bit replay.

For a submission, include these paths together:

- `solution.py`
- `feature_pipeline.py`
- `competition_data.py`
- `train_cv.py` (imported training utilities)
- `tsauc.py`
- `postprocess.py`
- the complete `detector/` directory
- `requirements.txt`
- the trained `model/` directory if using the pre-trained route

`build_submission_bundle.py` assembles exactly these files and writes a
SHA-256/size inventory. It excludes train data, feature caches, OOF arrays,
fold models, and rejected experiments.

The competition's current protocol deliberately hides the final online
horizon and supplies online values one at a time.  The implementation follows
that constraint; see the [official quickstarter](https://github.com/crunchdao/competitions/blob/master/competitions/structural-break-real-time/quickstarters/baseline/baseline.ipynb)
and CrunchDAO's [post-fix streaming clarification](https://forum.crunchdao.com/t/leaderboard-comparability-after-the-june-8-real-time-data-access-fix-were-pre-fix-scores-rescored/1188).

## Code map

- [`competition_data.py`](competition_data.py): parquet validation and
  zero-copy per-series views.
- [`feature_pipeline.py`](feature_pipeline.py): versioned bank composition,
  shared streaming state, sanitization, and resumable cache.
- [`tsauc.py`](tsauc.py): exact metric, accounting, fold SE, and training
  weights.
- [`train_cv.py`](train_cv.py): stratified series folds, deterministic row
  sampling, optional causal time regimes, LightGBM training, OOF prediction,
  uncertainty, importance, and final fit.
- [`ensemble_cv.py`](ensemble_cv.py): fixed and nested held-fold-excluded
  ensemble evaluation.
- [`stack_cv.py`](stack_cv.py): documented nested stacker experiment (rejected
  by CV, not deployed).
- [`solution.py`](solution.py): production `train`/`infer` entry points.
- [`postprocess.py`](postprocess.py): optional, causally decaying persistence
  for the absorbing break label.
- [`detector/`](detector): fully documented detector mathematics and each
  bank's independent offline/streaming self-test.
- [`tests/`](tests): metric equivalence, causality, numerical robustness,
  fold balance, reproducibility, and feature parity tests.
- [`validate_submission.py`](validate_submission.py): exact one-pass production
  model smoke test.
- [`build_submission_bundle.py`](build_submission_bundle.py): portable,
  checksummed deployment packager.

No finite validation result can guarantee first place on an unseen
leaderboard.  The defensible claim is narrower: the pipeline optimizes the
actual metric, eliminates known leakage paths, quantifies split uncertainty,
and leaves every modeling decision reproducible for the next iteration.
