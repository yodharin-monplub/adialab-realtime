# Structural Break: Real-Time — submission 3

This directory is a complete Crunch code submission. It trains a
feature-selected core/candidate LightGBM ensemble from scratch and then runs
strictly causal, one-value-at-a-time inference.

It contains no pretrained model weights, training observations, test
observations, cached feature matrices, or predictions. Before Crunch calls
`train()`, `resources/` is intentionally empty. The two NPZ files contain only
the frozen integer column IDs and matching names selected from the local
`train/` data.

Submission 3 is experimental. Its feature selection reduced the model inputs
from 2,658 to 1,439 columns, but its locked-test improvement over the
all-feature submission-2 baseline was only `+0.000138` TSAUC and was not
statistically resolved. See [FEATURE_SELECTION.md](FEATURE_SELECTION.md).

## Runtime lifecycle

```text
Crunch training iterator
        │
        ▼
main.train(...)
        │
        ▼
ensemble_training.train_ensemble(...)
        ├── validate and retain the one-pass training records
        ├── construct causal per-step targets
        ├── select a deterministic 20% row sample
        ├── calculate TSAUC-aligned row weights
        ├── build and project core features ─────► fit core LightGBM
        ├── build and project candidate features ► fit candidate LightGBM
        └── atomically publish models/manifests under resources/
                                                    │
                                                    ▼
Crunch test iterator ──► main.infer(...) ──► solution.infer(...)
                                                    │
                    for each newly revealed value ──┤
                    update causal detector state    │
                    select component columns        │
                    predict core and candidate      │
                    blend probabilities in logits   │
                                                    ▼
                                       yield one probability
```

## Submitted files

| Path | Runtime responsibility |
|---|---|
| `main.py` | Thin Crunch entry points and `INFER_PARALLELISM = 4`. |
| `ensemble_training.py` | Input validation, deterministic sampling, feature-contract validation, LightGBM fitting, and atomic resource publication. |
| `solution.py` | Model/manifest loading, selected-column mapping, streaming inference, and logit blending. |
| `feature_pipeline.py` | Canonical feature names/order plus matching offline and streaming feature interfaces. |
| `detector/*.py` | Individual causal detector banks. |
| `train_cv.py` | Shared deterministic sampler, TSAUC-aligned row weights, and LightGBM parameter contract. |
| `tsauc.py` | Local implementation of the competition's time-stratified AUC. |
| `postprocess.py` | Optional causal score persistence; disabled in this submission. |
| `core_feature_selection.npz` | Frozen ordered core `columns` and `names`; no weights or observations. |
| `candidate_feature_selection.npz` | Frozen ordered candidate `columns` and `names`; no weights or observations. |
| `resources/` | Writable model directory. Empty in the uploaded bundle and populated by `train()`. |
| `bundle_manifest.json` | Byte sizes and SHA-256 hashes for the submitted files. |

## Public API and tuple contracts

Crunch imports the top-level `main.py` and calls these functions:

```python
def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    ...

def infer(datasets: Iterable[tuple], model_directory_path: str):
    yield None                 # readiness sentinel
    yield probability_t0
    yield probability_t1
    ...
```

### Training records

Each training element must be:

```python
(dataset_id, historical, online, tau_index)
```

- `dataset_id`: unique integer series ID;
- `historical`: one-dimensional guaranteed-clean reference period;
- `online`: complete labeled period used only to reproduce causal training
  rows;
- `tau_index`: zero-based break index inside `online`, or `None`/a negative
  value for a no-break series.

For an online segment of length `L`, `_target()` constructs:

```python
target[t] = 1 if tau_index >= 0 and t >= tau_index else 0
```

The target is absorbing: after the break it remains one.

### Inference records

`solution._test_item_parts()` accepts either:

```python
(historical, online_iterator)
(dataset_id, historical, online_iterator)
```

`online_iterator` is consumed once. The implementation never converts the
complete online stream to an array, asks for its final length, or reads a
future observation.

## Training implementation

### 1. Entry point and record validation

`main.train()` delegates directly to:

```python
ensemble_training.train_ensemble(datasets, model_directory_path)
```

`_normalize_records()` consumes the one-pass iterator and rejects:

- tuples that do not contain four elements;
- duplicate dataset IDs;
- non-one-dimensional history or online arrays;
- empty periods;
- break indices outside the online segment.

The normalized records are retained because the core and candidate matrices
are constructed sequentially from the same input series.

### 2. Deterministic row sampling and TSAUC weights

The complete training set contains one supervised row for every online time.
To control peak memory, `_prepare_training_data()` retains a deterministic 20%
of those rows:

```python
TRAIN_SAMPLE_FRACTION = 0.20
TRAIN_SAMPLE_SEED = 99173
```

`train_cv.deterministic_sample_mask()` applies a fixed SplitMix64-style hash
to `(dataset_id, online_time, seed)`. Sampling is therefore reproducible and
independent of the label, Python's randomized hash, iteration order, and model
state.

Weights are calculated from the complete, unsampled class counts at each time
step. If `n_pos(t)` and `n_neg(t)` are the full training counts, then:

```text
positive row at t -> weight n_neg(t)
negative row at t -> weight n_pos(t)
```

Positive and negative contributions are balanced inside each time stratum,
while the total emphasis follows the competition's `n_pos(t) * n_neg(t)` pair
geometry. Nonzero weights are finally normalized to mean one.

### 3. Frozen component contracts

At module import, `_has_selected_feature_contract()` requires both selection
files or neither. Because both files are in this bundle, these specifications
are activated:

| Component | Source features | Selection | Model columns | Rounds | Weight |
|---|---:|---|---:|---:|---:|
| Core | 1,168 | gain/split/SHAP rank consensus | 768 | 101 | 0.55 |
| Candidate | 1,490 | gain ranking, then `abs(correlation) < 0.98` pruning | 671 | 217 | 0.45 |

`_feature_contract()` opens the appropriate NPZ with `allow_pickle=False` and
requires exactly two arrays:

```text
columns: int32 source-column vector
names:   matching ordered feature-name vector
```

It rejects empty, duplicate, negative, or out-of-range columns. It then
reconstructs every expected name from the submitted detector code and demands
exact ordered equality. A stale selection cannot silently point at a different
feature after a code change.

### 4. Causal feature materialization

For one component and one series, training calls:

```python
full = compute_features_offline(historical, online, specification.banks)
sampled_selected = full[series_sample_mask][:, selected_columns]
```

`full[t]` is permitted to depend on `historical` and `online[:t + 1]` only.
Independent regression tests compare this vectorized training path with the
incremental `CoreStreamDetector.step()` path and verify that modifying future
online values cannot change earlier feature rows.

Core source banks:

| Bank | Columns | Main signal family |
|---|---:|---|
| `d4` | 306 | Location, scale, robust shape, tails, quantile crossings, serial dependence, AR(5) residuals, and difference energy. |
| `d15` | 322 | Autocorrelation-adjusted EWMA evidence at multiple response speeds. |
| `d22` | 230 | Empirical historical-null calibration of suffix statistics. |
| `d26` | 184 | Empirical historical-null calibration of EWMA statistics. |
| `d27` | 70 | Empirical suffix calibration for extreme-tail and difference-energy specialists. |
| `d30` | 56 | Empirical EWMA calibration for the same specialist channels. |

Candidate source banks:

| Bank | Columns | Main signal family |
|---|---:|---|
| `d32` | 110 | Causal Haar-wavelet band energy and spectral shape. |
| `d33` | 112 | Deterministic MiniROCKET-style local rank/value patterns. |
| `d35` | 128 | Random-Fourier MMD on lag embeddings. |
| `d37` | 126 | GARCH forecast surprise and volatility-parameter scores. |
| `d40` | 324 | Empirically calibrated serial-copula transitions. |
| `d45` | 230 | Adjacent-block contrasts calibrated on clean history. |
| `d51` | 30 | Shiryaev–Roberts mean/scale change evidence. |
| `d52` | 64 | Higher-order historical-ECDF shape modes. |
| `d53` | 14 | Conditional AR-law generalized likelihood ratios. |
| `d55` | 132 | Conformal power-martingale evidence. |
| `d57` | 110 | Polynomial nonlinear forecast surprise. |
| `d58` | 110 | Random-Fourier nonlinear forecast surprise. |

Every output is converted to finite values and clipped to
`[-1_000_000, 1_000_000]` by `sanitize_features()`.

The two matrices are built and fitted sequentially, so the full core and
candidate training matrices are not resident at the same time.

### 5. LightGBM fitting

Both components use the parameter base in `train_cv.DEFAULT_MODEL_PARAMETERS`:

| Parameter | Value |
|---|---:|
| objective | `binary` |
| boosting | `gbdt` |
| learning rate | `0.035` |
| leaves | `31` |
| minimum rows per leaf | `300` |
| maximum bins | `127` |
| feature fraction | `0.60` |
| bagging fraction/frequency | `0.85` / `1` |
| L1/L2 penalty | `0.20` / `3.0` |
| threads | `min(8, available CPUs)` |
| deterministic/column-wise | `True` / `True` |

All LightGBM seeds are derived from `MODEL_SEED = 51413`. Cloud training uses
the fixed 101/217 round counts selected before the locked test was opened; it
does not perform early stopping on the data supplied to the final `train()`.

### 6. Resource publication

All temporary files are created with:

```python
TemporaryDirectory(prefix=".training_", dir=model_directory_path)
```

This matters because Crunch may mount the code directory read-only and allow
writes only under the supplied resources directory. Models are first staged,
checked for completeness, and then promoted with `os.replace()`. The top-level
manifest is replaced last, preventing inference from observing a new ensemble
manifest with missing component files.

After a successful call, the resource tree is:

```text
resources/
├── manifest.json
├── core/
│   ├── final.txt
│   └── manifest.json
└── candidate/
    ├── final.txt
    └── manifest.json
```

Each component manifest records its banks, selected names, source column IDs,
source feature count, selection method/hash, sample settings, model parameters,
round count, training-row count, and model seed. The top manifest records the
relative component directories, weights, logit blend method, and selection
contract hash.

## Streaming inference implementation

### 1. Load and validate resources

`main.infer()` delegates to `solution.infer()`, which calls `_load_ensemble()`.
For each component, `_load_single_component()` verifies:

- the manifest exists;
- feature names are nonempty and unique;
- every declared name still exists in the specified detector banks;
- source column IDs are unique, in range, and reproduce the declared names;
- a model file exists;
- the LightGBM model's feature count equals the manifest feature count.

The loader constructs the union of the core and candidate banks once. It maps
each component's selected names into the union detector row by name, preserving
the exact ranked training order.

### 2. Readiness sentinel

The first value from the generator is deliberately `None`:

```python
generator = infer(test_iterator, resources_path)
assert next(generator) is None
```

Crunch uses this as the readiness signal. Every subsequent yield is a model
score for exactly one newly released online value.

### 3. One independent detector per series

For each test series:

```python
detector = CoreStreamDetector(historical, union_banks)

for value in online_iterator:
    union_row = detector.step(float(value))
```

The clean historical period initializes frozen reference distributions,
scales, AR fits, empirical nulls, and other detector state. `step()` advances
all active states exactly once. State is discarded at the end of the series,
so no information crosses between series.

### 4. Component prediction and logit blend

At every time step, inference projects the union row into the core and
candidate column orders and calls each LightGBM with `num_threads=1`.
If `p_core` and `p_candidate` are the probabilities, the final value is:

```text
z = 0.55 * logit(p_core) + 0.45 * logit(p_candidate)
p = sigmoid(z)
```

Probabilities are clipped to `[1e-7, 1 - 1e-7]` before conversion to logits.
`postprocess_penalty` is `None`, so `StreamingDecayedMax` returns the raw blend
without persistence smoothing. Non-finite results fall back to `0.5`, followed
by final clipping to `[1e-7, 1 - 1e-7]`.

`INFER_PARALLELISM = 4` tells Crunch that four separate processes may handle
disjoint series. Each individual LightGBM prediction remains single-threaded,
which avoids nested CPU oversubscription.

## TSAUC alignment

The competition compares predictions only among series observed at the same
online time. For time `t`, let `AUC(t)` be that cross-series ROC AUC. The metric
implemented in `tsauc.py` is:

```text
TSAUC = sum_t n_pos(t) n_neg(t) AUC(t)
        --------------------------------
              sum_t n_pos(t) n_neg(t)
```

This is why training uses time-specific opposite-class weights and why a
pure elapsed-time feature cannot improve the score: all rows compared inside
one AUC stratum have the same elapsed time.

## Feature-selection provenance and leakage controls

Feature selection is not rerun in the Crunch container. The two NPZ contracts
were frozen by the parent repository's `feature_selection_experiment.py` using
only the supplied `train/` folder:

- folds 2–4: ranking/model fitting, 4,800 whole series;
- fold 1: method, subset-size, round-count, and blend selection, 1,600 series;
- fold 0: locked test, 1,600 series;
- no series appears in more than one role;
- gain, split count, SHAP, and correlation rankings used training rows only;
- the random-256 negative control was ineligible for deployment;
- the locked test was scored once after `frozen_selection.json` was written.

The selected ensemble scored `0.621166` on validation and `0.630597` on the
locked test. The all-feature baseline scored `0.614042` and `0.630459`. The
paired five-subfold improvement had SE `0.001653` and a 95% t interval of
`[-0.004187, 0.004994]`; the interval includes zero.

Contract hash:

```text
e7154880585cb472bbed94079048312641d5b4f03d6dfc912ac4a8e908861a8f
```

## Determinism and causal invariants

The implementation is designed around these invariants:

1. A score at online time `t` uses only clean history and values `0..t`.
2. Changing values after `t` cannot change the score at `t`.
3. Whole series, rather than individual rows, define offline data splits.
4. Feature selection files contain metadata only, not fitted model parameters.
5. Training row selection is a deterministic, label-independent hash.
6. Feature order is checked by both integer column and feature name.
7. Model seeds and detector random-feature seeds are fixed.
8. No state is shared across test series.
9. All runtime writes remain below `model_directory_path`.
10. The top-level model manifest is published only after both models exist.

The parent repository's regression suite checks offline/streaming feature
parity, future-value invariance, finite constant-series behavior, deterministic
sampling, exact TSAUC behavior, selected-column ordering, and ensemble loading.
The bundle has also completed a read-only-code/writable-resources smoke test
covering `train()` followed by `infer()`.

## Running and submitting

Install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

For the Crunch CLI, initialize the project using the command shown on the
competition's **Submit via CLI** page, copy this directory's contents into that
project, and run:

```bash
crunch test
crunch push --message "submission 3 selected core candidate"
```

For the web uploader, upload this directory as the submission root. Do not
upload the parent repository, local `train/`, local test labels, feature
caches, or experiment artifacts.

### Direct local API use

The bundle uses top-level imports because Crunch makes the submitted directory
the Python code root. If calling it from a parent directory, insert the bundle
path before importing `main`:

```python
from pathlib import Path
import sys

submission = Path("/home/hp/resources/submission_3").resolve()
sys.path.insert(0, str(submission))

from main import infer, train

train(training_tuple_iterator, str(submission / "resources"))

prediction_generator = infer(test_tuple_iterator, str(submission / "resources"))
assert next(prediction_generator) is None
predictions = list(prediction_generator)
```

Do not use `from submission_3.main import ...` unless every internal import is
converted to package-relative form; the uploaded Crunch layout intentionally
uses `from ensemble_training import ...` and similar top-level imports.

## Common failures

| Error | Meaning and correction |
|---|---|
| `ModuleNotFoundError: ensemble_training` | The bundle was imported as a package from its parent. Put `submission_3/` itself at the front of `sys.path`, or run through Crunch. |
| `missing trained-model manifest` | `infer()` was called before `train()` successfully populated `resources/`. |
| `PermissionError` under the code directory | Pass a writable `model_directory_path`; temporary and final files are designed to live only there. |
| `incomplete selected-feature bundle` | One of the two `*_feature_selection.npz` files is missing. Restore both files. |
| `feature columns and names describe different contracts` | Detector code, selection NPZ, or a component manifest was modified without rebuilding the contract. |
| `sampled training rows must contain both target classes` | A small local smoke dataset produced only one sampled class; use a representative labeled sample. |
| History-length error from a candidate detector | Some candidate banks require at least 529 clean history points. Competition histories are within the supported range. |

## What may be changed safely

- Changing `INFER_PARALLELISM` changes process-level throughput, not scores.
- Changing model rounds, selected columns, bank order, weights, detector math,
  LightGBM parameters, or postprocessing changes the model contract and
  requires new leakage-safe validation.
- If either selection NPZ is regenerated, update its names, the frozen contract
  hash, validation report, bundle manifest, and both training/inference tests.
- Keep `resources/` empty in the uploaded source bundle when a fresh cloud
  training run is required.
