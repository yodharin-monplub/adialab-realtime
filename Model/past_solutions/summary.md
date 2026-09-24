# Previous ADIA Lab Structural Break winners: methods and transfer audit

## Scope and status legend

This document audits the material in `models/past_solutions/` and the public page for
the 2025 ADIA Lab Structural Break Challenge against the current real-time
solution and the feature-selection study completed after `submission_2`.

Status markers:

- **✅ Current** — used in the deployed `submission_2` feature/model contract.
- **🧪 Tried** — implemented and evaluated here, but rejected or not deployed.
- **◐ Partial** — a causal analogue exists, but not the winner's exact method.
- **❌ Not tried** — no corresponding experiment was found in this repository.
- **🚫 Rewrite** — the old formulation uses the complete post-boundary period
  and would leak future observations in the real-time competition. It must be
  rewritten as history-versus-current-prefix/suffix evidence.

“Tried” refers to the current real-time repository, not merely to its presence
inside the archived winning solutions.

## Why the old and new tasks are different

The [old competition page](https://hub.crunchdao.com/competitions/structural-break)
and [official documentation](https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge)
describe a fixed-boundary classification problem: the potential boundary is
given, the complete series before and after it is available, one probability
is emitted per series, and the metric is ordinary ROC AUC.

The real-time edition reveals period 2 one observation at a time. A valid
prediction at online time `t` may use the clean history and online prefix
`0..t`, but not `t+1..end`. Its TSAUC compares series only within the same
online time. Consequently:

- full pre/post tests from the old challenge cannot be copied directly;
- boundary-distance and “post head of fixed final length” features are illegal;
- valid adaptations compare the frozen history with the currently available
  suffix or maintain causal sequential evidence such as CUSUM/EWMA;
- elapsed time alone cannot add TSAUC discrimination within a time stratum.

The [open-benchmark documentation](https://docs.crunchdao.com/competitions/competitions/structural-break-open-benchmark)
records the old top score as 90.14% ROC AUC and identifies the first-place team
as Humberto Brandão, Mário Augusto Filho, Rafael Alencar, and João Pedro
Peinado. The [CrunchDAO forum](https://forum.crunchdao.com/t/winning-solutions-of-structural-break-challenge/1070)
states that winner write-ups were published through the leaderboard. Three
such solution packages are present locally.

## Archive solution 1: TabPFN + selected LightGBM ensemble

Sources: `models/past_solutions/solution 1/README.md` and
`models/past_solutions/solution 1/submission.ipynb`.

### Feature transformations and tests

| Winner method | Current status | Audit note |
|---|---|---|
| Raw values | ✅ Current | Raw observations feed all causal detectors. |
| Per-series z-score | ✅ Current | Frozen history mean and SD are used. |
| Cumulative sum | ✅ Current | Prefix sums and CUSUM scans are central features. |
| Dense-rank transform | ✅ Current | Frozen historical ECDF ranks are used instead. |
| Absolute value | ✅ Current | Robust-scale and absolute-residual channels are used. |
| Rolling mean, window 16 | ◐ Partial | Causal standardized suffix sums cover windows including 16. |
| Rolling standard deviation, window 16 | ◐ Partial | Scale/energy suffix channels cover this signal, not the identical formula. |
| Mean, median, SD, min, max, skew summaries | ◐ Partial | Mean/scale/rank-shape evidence is present; not every raw summary is exposed. |
| Mean/SD and median/SD ratios | ◐ Partial | Standardized location evidence exists, but not these exact static ratios. |
| Lag-1 correlation | ✅ Current | Multiple raw, rank, volatility, and residual dependence channels are used. |
| Quantiles 25/50/75 | ◐ Partial | Quantile-crossing rates and ECDF modes are used rather than raw quantile estimates. |
| Boundary windows 20/60/120/500/1000 | 🚫/◐ Partial | Current suffix windows 4–512 are causal; the complete post-head formulation is invalid online. |
| F-test for variance | 🧪 Tried now | `d60` tests a history-calibrated causal log-variance/F-like suffix statistic; it deliberately avoids an uncalibrated textbook p-value. |
| Levene variance test | 🧪 Tried now | `d60` adds Brown–Forsythe/Levene-like absolute-deviation suffix evidence calibrated on clean-history pseudo-suffixes. |
| Two-sample Kolmogorov–Smirnov test | 🧪 Tried | Binned/empirical KS variants were implemented but excluded from submission 2. |

### Feature selection and modeling

| Winner method | Current status | Audit note |
|---|---|---|
| Five-fold OOF TabPFN probabilities as a feature | ❌ Not tried | Submission 2 has no TabPFN component. |
| SHAP top 200 and top 500 | 🧪 Tried now | The new study tested SHAP top 64/128/256/512/768 separately for both families. |
| LightGBM gain top 200 and top 500 | 🧪 Tried now | Gain subsets and nonzero-gain subsets were tested. |
| Four LightGBMs over SHAP/gain subsets | ◐ Partial | Separate subset models were tested; the exact four-model average was not. |
| Seed diversity/model averaging | ◐ Partial | Deterministic seeds and a two-family ensemble are used, not this four-seed design. |
| Top-level average of model probabilities | ✅ Current | Submission 2 blends two models in logit space rather than arithmetic probability space. |

The archived notebook appears to calculate SHAP and gain importance on the
full labeled training matrix before fitting its final ensemble. For the new
real-time study, rankings were restricted to the training split, selection
was performed on a separate validation split, and fold 0 stayed locked until
the selection contract was frozen.

## Archive solution 2: broad statistical-test and signal-feature bank

Source: `models/past_solutions/solution 2/README.md`.

| Winner method | Current status | Audit note |
|---|---|---|
| Raw series, pre/post segments, and returns | 🚫/◐ Partial | Raw/history-prefix features exist; complete post segments are unavailable online; percentage returns were not used. |
| Mean, SD, kurtosis, skewness, q10/q25/q75 | ◐ Partial | Location, scale, historical kurtosis, and ECDF Legendre shape modes are used. |
| Signal-to-noise `abs(mean)/SD` | ◐ Partial | Standardized prefix/suffix location scores are the causal analogue. |
| Rolling mean/SD/kurtosis/skew | ◐ Partial | Multi-window location/scale/shape scans exist, but not every exact rolling statistic. |
| Lag-1 autocorrelation | ✅ Current | Raw, rank, volatility, and residual dependence changes are represented. |
| Linear trend slope | 🧪 Tried | OLS slope/t-stat suffix features were implemented in an earlier detector and rejected. |
| Local windows ±7/15/30/100 | 🚫/◐ Partial | Adjacent-block and suffix tests are causal analogues; symmetric future windows are illegal. |
| Local SD difference | ✅ Current | Scale, difference-energy, GARCH surprise, and adjacent-block scale signals are used. |
| Levene test | 🧪 Tried now | `d60` includes the causal Brown–Forsythe/Levene-like adaptation; a full future-bearing pre/post test remains illegal. |
| Sliding variance-ratio dynamics | 🧪 Tried now | `d60` adds six rolling log-variance/F-like channels on top of the existing scale, energy, and GARCH evidence. |
| Distance from detected change to supplied boundary | 🚫 Rewrite | The current task does not supply the true break time at inference. |
| Hurst exponent | 🧪 Tried now | `d60` estimates rolling roughness from lag-1/2/4/8 increment energies at four causal window lengths. |
| Kolmogorov–Smirnov | 🧪 Tried | Implemented in rejected distribution banks. |
| Anderson–Darling | 🧪 Tried now | `d61` adds a frozen-quantile-grid causal AD-like tail-weighted discrepancy. |
| Cramér–von Mises | 🧪 Tried | A binned CvM-like statistic was implemented in an early bank. |
| Mood median test | 🧪 Tried now | `d60` uses a history-calibrated causal median-side occupancy analogue at six suffix lengths. |
| Fligner–Killeen variance test | 🧪 Tried now | `d60` rank-normalizes absolute deviations and calibrates their causal rolling mean on clean history. |
| Energy distance | 🧪 Tried now | `d61` adds a discretized causal energy-distance analogue over frozen historical bins. |
| Daubechies-4 wavelet decomposition | 🧪 Tried now | `d60` adds four causal stationary db4 detail-energy levels at windows 32/128/512. |
| Sample entropy | 🧪 Tried now | `d63` adds guarded causal sample-entropy evidence at multiple suffix scales. |
| AR(5) coefficients | ✅ Current | Regularized Yule–Walker AR(5), residuals, scores, and AR GLRs are used. |
| `tsfresh` features | ❌ Not tried | No `tsfresh` extraction was used. |
| CatBoost | ❌ Not tried | Current supervised models are LightGBM. |
| XGBoost | ❌ Not tried | Current supervised models are LightGBM. |
| LightGBM | ✅ Current | Two independent GBDT classifiers form the final ensemble. |

## Archive solution 3: Alphabot first-place solution

Source: `models/past_solutions/solution 3/Alphabot_Solution_StructuralBreak_ADIALab.pdf`.

The central design was diversity: four researchers independently built
feature/model families, eight first-level tree models generated predictions,
and a second-level stacker combined them with selected statistical features.

### Humberto: 58 statistical and structural-break features

The table preserves every named feature idea in the report, while grouping
closely related parameter variants.

| Report features | Idea | Current status |
|---|---|---|
| f01, f20 | Local Welch t-tests on absolute values, windows 50/100 | 🧪 Tried now as history-calibrated absolute-magnitude suffix evidence over windows 16–512; not the future-bearing original test |
| f02, f36 | Fisher aggregation of multi-window Welch p-values | 🧪 Causal analogue tried in `d65`, which combines multi-window detector evidence without labels |
| f03 | Distance between a CUSUM/MOSUM change estimate and known boundary | 🚫 Rewrite; CUSUM itself is ✅ Current |
| f04, f22 | Piecewise cumulative-path regression; gain from adding a step | 🧪 Tried now through fixed-age causal step, scale, slope, and piecewise-regression scans in `d62` |
| f05, f32 | Fligner–Killeen dispersion p-values | 🧪 Tried now through rank-normalized absolute-deviation suffix evidence in `d60` |
| f06, f35 | Robust/in-band pre/post slope differences | 🧪 Partial slope tests tried; exact in-band version not tried |
| f07, f25, f42 | Central-slab or full Shannon entropy differences | ◐ Wavelet entropy deployed; exact distribution entropy not tried |
| f08, f41, f56 | Jensen–Shannon divergence on positive/central/shape distributions | 🧪 Tried in early distribution bank; not deployed |
| f09, f23, f52 | F variance tests and Fisher aggregation | 🧪 Partial now: six causal log-variance/F-like channels; Fisher aggregation remains untried |
| f10, f16, f47 | Tail/ring occupancy around historical mean and SD bands | 🧪 Tried now: five in-band widths and four asymmetric expected-shortfall levels; the exact ring grid remains absent |
| f11, f18 | Hellinger distance on positive/central support | 🧪 Tried now over 16 frozen historical-quantile bins and three causal suffix lengths |
| f12 | BDS independence/nonlinearity test | 🧪 Tried now through symbolic transition/collision and BDS-like evidence in `d63` |
| f13 | Supremum-F step/hinge breakpoint scan | 🧪 Tried now through guarded fixed-age causal scans in `d62` |
| f14, f38 | Minimum p-value over robust local two-sample comparisons | ❌ Not tried |
| f15, f17, f24, f26, f33, f48, f57 | Near-zero/in-band occupancy or absolute-deviation differences at multiple widths | 🧪 Tried now with five standardized bands over windows 16/64/256 and history-null calibration |
| f19 | Mean absolute deviation inside a tight zero band | ❌ Not tried |
| f21, f54 | Trimmed skewness/kurtosis | ◐ ECDF Legendre P3/P4 shape channels are causal analogues |
| f27, f30, f49 | Coefficient-of-variation and pre/post CV differences | ◐ Standardized scale evidence exists; exact CV features absent |
| f28 | Core-window ring occupancy difference | ❌ Not tried |
| f29, f44 | Signed distributional AUC/G-statistic contrasts | ❌ Not tried |
| f31 | Best variance/change ratio across candidate windows | ◐ Multi-window maxima are used; exact ratio absent |
| f34, f37 | Spearman correlation of absolute magnitude with time | 🧪 Tried now through frozen-rank magnitude/time trend channels in `d64` |
| f39 | CUSUM-of-squares at the known boundary | ◐ Scale CUSUM is deployed; fixed-boundary statistic requires rewrite |
| f40 | Log variance of trimmed absolute values | ◐ Robust scale/log-SD signals exist; exact statistic absent |
| f43 | Historical 35th percentile | ◐ Frozen historical ECDF/quantiles are used, not this standalone value |
| f45 | Shape-standardized Wasserstein-1 distance | 🧪 Tried in an early binned distribution bank |
| f46, f55 | PACF/ACF equality p-values with Fisher transforms | 🧪 Tried now through history-calibrated Fisher ACF/PACF shift evidence in `d64` |
| f50 | Historical first-half SD | ◐ Historical scale and calibration descriptors are used |
| f51 | Mean run length in a narrow band | 🧪 Partial now: current causal in-band run length is included at five widths; historical mean run length is used through calibration rather than as a separate column |
| f53 | Minimum Wilcoxon p-value | 🧪 Causal analogue tried with multi-window frozen-rank Wilcoxon evidence and maxima in `d61` |
| f58 | Change in 95% expected shortfall | 🧪 Tried now: lower/upper tail-excess means at 1%, 2.5%, 5%, and 10%, calibrated against clean history |

### Mário: hierarchical descriptive, financial, divergence, and evolved features

| Winner method | Current status | Audit note |
|---|---|---|
| Mean, SD, min, max, skew, kurtosis, q10–q90 | ◐ Partial | Most change aspects exist, but not the full static summary grid. |
| ACF lags 1/2/5/10/20 | ◐ Partial | Lags 1/2/3/5 plus AR and copula dependence are used. |
| Sharpe and Sortino ratios | ❌ Not tried | No financial-return ratio features. |
| Profit factor | ❌ Not tried | No positive/negative return-sum ratio. |
| CVaR at 1/2.5/5/10% | 🧪 Tried now | `d60` adds lower and upper expected-shortfall excess at all four levels over three causal windows. |
| Regression R², slope, intercept | 🧪 Partial | Slope and AR fit descriptors were tried; full grid absent. |
| Gini ratio | ❌ Not tried | No Gini feature. |
| Maximum drawdown | ❌ Not tried | No drawdown feature. |
| Pre/post difference, absolute difference, product, ratio | 🚫/◐ Partial | Causal deviations and ratios exist; complete post comparisons need rewrite. |
| KS statistic/p-value/location/sign | 🧪 Tried | KS variants were rejected. |
| Wasserstein distance | 🧪 Tried | Binned W1 was rejected. |
| Euclidean distance between clipped distributions | ❌ Not tried exactly | MMD is a related deployed joint-distribution distance. |
| 287 validation-evolved composite features | ❌ Not tried | No genetic/random feature-expression search was run. |
| `sin`, `cos`, `tanh`, `log`, `exp`, square root, sign interactions | ◐ Partial | Cosine random Fourier features, logs, absolute values, and polynomial bases exist; not the evolved formulas. |
| Raw, absolute, log1p, cumulative sum, cumulative product transforms | ◐ Partial | Raw/absolute/cumulative/log context are used; cumulative product is absent. |
| Full and boundary 10% slices | 🚫/◐ Partial | Current multi-scale suffixes are the legal causal adaptation. |

### Rafael: robust transformations, distances, and generated composites

| Winner method | Current status | Audit note |
|---|---|---|
| IQR filtering with multipliers 1.0/1.75/2.0 | 🧪 Tried now | `d66` uses frozen-history IQR fences at 1.0/1.75/2.0 with winsorized moments and outlier rates. |
| Absolute and sign-flipped signal variants | ◐ Partial | Absolute channels are common; explicit sign-flipped duplicate absent. |
| Symlet-2 level-1 wavelet denoising | 🧪 Causal analogue tried | `d66` adds stationary Symlet-2 detail-energy channels while avoiding noncausal boundary reconstruction. |
| Cumulative sum | ✅ Current | Prefix/CUSUM evidence deployed. |
| Percentage change | ❌ Not tried | Lag difference energy is used, not percentage returns. |
| Global/windowed mean, median, extrema, SD, variance, mode, percentiles | ◐ Partial | Causal multi-window moments/rank-shape evidence cover a subset. |
| Window sizes 20–240 on pre-tail/post-head | 🚫/◐ Partial | Similar causal suffix scales exist; fixed post head leaks online. |
| Canberra distance | 🧪 Tried now | `d60` compares causal suffix and frozen-history histograms. |
| Bray–Curtis distance | 🧪 Tried now | `d60` compares causal suffix and frozen-history histograms. |
| Hamming distance | ❌ Not tried |
| Jensen–Shannon distance | 🧪 Tried, rejected |
| Rogers–Tanimoto distance | ❌ Not tried |
| Dice distance | ❌ Not tried |
| Yule distance | ❌ Not tried |
| Cityblock/Manhattan distance | ❌ Not tried |
| Full/windowed/within-history KS tests | 🧪 Partial | KS and historical empirical-null calibration were tried; exact grid absent. |
| 477 main + 22 alternate engineered composites | ❌ Not tried | Tree interactions are learned, but formulas were not explicitly generated. |
| Arithmetic, powers, square root, `tanh`, modulo, integer division | ❌ Not tried as an evolved composite search |

### João and the ensemble

| Winner method | Current status | Audit note |
|---|---|---|
| Combine up to six first-level predictions/statistics | 🧪 Partial | A nonlinear stacker over core/candidate predictions was tried and rejected. |
| Subtraction, multiplication, division | ❌ Not tried as explicit stack features |
| Arithmetic/geometric/harmonic means | ❌ Not tried |
| Eight first-level XGBoost/RF models | ❌ Not tried | Current first level is two LightGBM models. |
| XGBoost classifier and regressor variants | ❌ Not tried |
| Five-seed bagging | ❌ Not tried exactly |
| Random Forest | ❌ Not tried |
| Second-level stacking to about 100 features | 🧪 Partial | A 13-feature nested LightGBM stacker scored worse than equal logit blending. |

## Current methods not inherited directly from the old winners

Submission 2 also introduced methods not clearly present in these archived
solutions:

- empirical-null conversion of per-series statistics to Gaussian scores;
- autocorrelation-adjusted, finite-start EWMA evidence;
- Haar spectral entropy/centroid/slope/peak;
- causal MiniROCKET pattern features;
- random-Fourier MMD on lag embeddings;
- GARCH parameter-score and forecast-surprise channels;
- complete coarse serial-copula transitions;
- Shiryaev–Roberts sequential likelihood evidence;
- conditional AR generalized likelihood ratios;
- conformal power martingales;
- polynomial and random-Fourier nonlinear forecast-surprise detectors;
- exact TSAUC-aligned row weights and time-stratified evaluation.

## Feature-selection experiment completed after submission 2

The leakage-safe driver is `feature_selection_experiment.py`; machine-readable
outputs are under `models/artifacts/feature_selection_v1/`.

### Split protocol

- Train: folds 2–4, 4,800 complete series and 2,408,089 rows.
- Validation: fold 1, 1,600 complete series and 808,302 rows.
- Locked test: fold 0, 1,600 complete series and 803,393 rows.
- Rankings used train rows only.
- Validation selected methods, subset sizes, and ensemble weights.
- The locked test was opened only after `frozen_selection.json` was written.

### Selection methods tested independently for core and candidate

- all features;
- LightGBM gain top 64/128/256/512/768;
- LightGBM split-frequency top 64/128/256/512/768;
- train-only mean-absolute-SHAP top 64/128/256/512/768;
- consensus of gain, split, and SHAP ranks at the same sizes;
- all nonzero-gain features;
- gain ranking followed by absolute-correlation pruning at 0.98;
- whole-bank cumulative selection;
- a fixed random-256 negative control, excluded from deployment selection.

### Results

| Configuration | Validation TSAUC | Locked-test TSAUC |
|---|---:|---:|
| All-feature 50/50 baseline | 0.614042 | 0.630459 |
| Selected: core consensus-768 + candidate gain/correlation-671, 55/45 | 0.621166 | 0.630597 |
| Difference | +0.007124 | +0.000138 |

The paired five-subfold test difference has SE `0.001653` and a 95% t interval
of `[-0.004187, 0.004994]`. Feature selection therefore produced no
statistically resolved held-out improvement. It did, however, reduce the two
model inputs from 2,658 to 1,439 total columns and may still be useful as a
lower-cost external cloud experiment.

## Single joint LightGBM study with newly adapted winner features

The follow-up driver is `joint_lightgbm_experiment.py`; full methodology is in
`JOINT_LIGHTGBM.md` and the sealed report is in
`joint_lightgbm_results.md`. It concatenated the 1,168 core, 1,490 candidate,
and 129 new `d60` columns, then performed one global selection rather than
training a model per family.

### Leakage-control protocol

- Fresh seed `20260821`, grouped by complete series: 4,800 train, 1,600
  validation, and 1,600 sealed-test series.
- Repeated LightGBM rankings and correlation pruning used train only.
- Subset choice, ten fixed-seed hyperparameter candidates, early stopping, and
  the core/candidate blend weight used validation only.
- Columns, parameters, iterations, and baseline were hash-frozen before the
  single held-out scoring pass.
- Historical work had already examined this repository's training population,
  so this is a procedurally leakage-controlled split, not a claim of a never-
  before-seen external population.

### Feature-selection and tuning results

| Development configuration | Validation TSAUC |
|---|---:|
| Existing 2,658 features, one LightGBM | 0.611450 |
| Expanded 2,787 features, one LightGBM | 0.612879 |
| Balanced global top 768, default LightGBM | 0.615989 |
| Balanced global top 768, tuned 95-leaf LightGBM | **0.616458** |
| Same-split separate core/candidate incumbent | 0.615903 |

The selected 768-column contract contains 304 core, 387 candidate, and 77
`d60` features: 28 robust scale/location, 22 expected-shortfall, 14 in-band/run,
7 db4, 5 distribution-distance, and 1 Hurst column. The new bank therefore
survived the frozen family quota and the joint model narrowly won development
validation by `+0.000555` over the separate-model incumbent.

### One-shot sealed result

| Frozen model | Pooled TSAUC | Series-bootstrap SE | Bootstrap 95% interval |
|---|---:|---:|---:|
| Single selected joint LightGBM | 0.614092 | 0.010389 | [0.593701, 0.632694] |
| Separate core/candidate incumbent | **0.615405** | 0.010577 | [0.594047, 0.634212] |

The pooled joint-minus-incumbent difference was `-0.001313`. Its paired
complete-series bootstrap SE was `0.005772` with a 95% interval of
`[-0.012181, 0.009776]`; the five disjoint series-group estimate likewise had
SE `0.005296` and a 95% t interval of `[-0.015454, 0.013952]`. Thus the apparent
validation advantage reversed and is statistically unresolved. The
precommitted submission gate failed, so no `submission_5` folder was created
and no post-test retuning is permitted.

## Winner ideas added for the joint-LightGBM study

The new `d60` bank in `detector/candidate/detectors60.py` implements 129 causal columns drawn from ideas that were
marked untried above. It does not copy the old complete pre/post formulas. At
online time `t`, every value uses only clean history plus prefix `0..t`, and
its reference center/scale is fitted only on equal-length pseudo-suffixes from
the clean history.

- Six-window log-variance/F-like, Brown–Forsythe/Levene-like,
  absolute-magnitude Welch-like, rank-Fligner, and median-side Mood-like
  evidence.
- Hellinger, Canberra, and Bray–Curtis distances between a causal suffix
  histogram and 16 frozen historical-quantile bins.
- Lower/upper expected-shortfall excess at 1%, 2.5%, 5%, and 10%; five in-band
  occupancy widths; and causal current-run lengths.
- Lag-energy Hurst/roughness estimates and four-level causal stationary db4
  detail energy.
- Cross-window max-absolute and signed-max reductions for the strongest
  evidence at each online step.

Independent vectorized-offline and stateful-streaming implementations are
covered by parity and future-causality tests. Inclusion in the bank means
“tried”; it does not mean LightGBM selected the column or that it improved the
sealed score. The joint study records those outcomes separately.

## Model 8 follow-up on remaining winner ideas

Model 8 added six causal banks (`d61`-`d66`, 239 columns) covering rank and
AD-like distribution shifts, discretized energy distance, Sup-F/piecewise
scans, sample entropy/BDS-style dependence, Fisher ACF/PACF changes,
label-free evidence aggregation, IQR robustness, and stationary Symlet-2
energy. All 49 repository detector sources passed independent streaming
parity and synthetic causality audits.

Across five complete-series development folds, `d61` was the only bank to
pass the predeclared above-chance and robust complementarity rule. The fixed
all-six nested blend scored 0.621595 ± 0.005260 versus 0.622097 ± 0.004856 for
the Model 6 candidate baseline. Its complete-series bootstrap SE was 0.005712
with a 95% interval of [0.608887, 0.630021]. The 0.64 target was not reached,
and the outer holdout was not used during this development phase.

The later user-requested frozen outer evaluation scored 0.605003 for the 95/5
Model 6/Model 8 blend versus 0.604658 for the candidate baseline, a +0.000345
change. Its complete-series bootstrap SE was 0.013386 with a 95% interval of
[0.579009, 0.633111]. No choice changed after scoring, but this was not a
pristine test because Model 6 had previously opened the same split. See
`models/model8/README.md` for the full detector and architecture tables.

## Highest-priority ideas still untried

1. Exact (rather than discretized) energy-distance suffix evidence with a
   computationally tractable streaming update.
2. Minimum-p corrections that explicitly model dependence among overlapping
   windows; Model 8 currently exposes maxima and label-free combinations.
3. Validation-evolved arithmetic/composite features with a nested search that
   prevents expression selection from seeing the sealed split.

These remain research candidates. Every future adaptation must retain exact
offline/streaming parity and use only the prefix available at the current
online step.
