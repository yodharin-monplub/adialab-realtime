# Joint LightGBM + new causal feature study

## What this experiment answers

The repository previously trained independent LightGBMs on 1,168 core and
1,490 candidate columns. A naïve model over their 2,658-column concatenation
scored 0.626153 on the old fold 0, below the separate blend's 0.633718. This
study tests the missing hypothesis: one joint learner **after global feature
selection**, with a separate bank of previously untried winner-inspired
features.

The executable driver is `joint_lightgbm_experiment.py`. It reads only aligned
caches generated from `train/`; `protected.zip` and competition test data are
never opened.

## New d60 feature contract

`detector/candidate/detectors60.py` contributes 129 features. The old competition's complete
pre/post calculations were rewritten for real-time legality: at online step
`t`, every value uses only the clean history and online prefix `0..t`.

- F variance-ratio, Brown–Forsythe/Levene-like absolute-deviation,
  absolute-Welch, Fligner, and Mood-median evidence over causal suffixes.
- Hellinger, Canberra, and Bray–Curtis distances over 16 historical-quantile
  bins.
- Lower and upper expected-shortfall excess at 1%, 2.5%, 5%, and 10% tails.
- In-band occupancy and current run lengths at five standardized widths.
- Rolling Hurst/roughness estimates from lag 1/2/4/8 increment energies.
- Four-level causal stationary Daubechies-4 detail energies.

Every statistic is converted to a within-series score using equal-length
pseudo-suffixes from the guaranteed-clean history. `compute_features_offline`
is vectorized for cache construction; `StreamDetector60` is independent and
incremental for cloud inference. Regression tests require their values to
match and verify that changing future online values cannot affect earlier
rows.

## Leakage boundary

The fresh deterministic split uses seed `20260821` and assigns complete
series—not rows—to three roles:

- folds 2, 3, 4: ranking and model fitting (60%);
- fold 1: subset, hyperparameter, early-stopping, and blend-weight selection
  (20%);
- fold 0: sealed one-shot test (20%).

The driver enforces these stages:

1. `screen`: three fixed-round train-only LightGBMs generate gain/split
   consensus ranks; correlation pruning also uses train rows only.
2. `select`: low-fidelity successive screening tests feature counts
   128–1,536, then reevaluates a shortlist at the full 20% row sample.
3. `tune`: ten deterministic LightGBM configurations are compared on
   validation only. Every configuration uses the same model/randomization
   seed, so the comparison isolates hyperparameters instead of seed luck.
4. `baseline`: the submission-3 core/candidate contracts are retrained on the
   same split and their logit weight is selected on validation.
5. `freeze`: selected columns, parameters, rounds, and the baseline are hashed
   before test targets may be used for scoring. Labels are used only to create
   a balanced stratified series split; no test score, prediction comparison,
   ranking, or model choice is made before the freeze.
6. `evaluate-test`: all three frozen deployment models are refit on
   train+validation, label-free test predictions are saved, and only then are
   test labels opened once.

The standard error is the sample SD of five deterministic disjoint test-series
group TSAUCs divided by `sqrt(5)`. A complete-series bootstrap and paired
joint-minus-incumbent uncertainty are also reported. This avoids pretending
that millions of rows from the same 1,600 series are independent.

## Commands and resumability

```bash
python3 train_cv.py build-cache \
  --train-directory train \
  --cache-directory models/artifacts \
  --cache-name novel_features \
  --banks d60 \
  --workers 8

python3 joint_lightgbm_experiment.py development
python3 joint_lightgbm_experiment.py evaluate-test --bootstrap-replicates 200
```

Completed screening/training jobs are reused only when their configuration
hash matches. `evaluate-test` refuses to overwrite a completed report. A
sentinel is written immediately before the one allowed held-out scoring pass.

An earlier concurrent run froze `frozen_contract.json` after varying the seed
between tuning candidates. Its test was never opened and that file is retained
only as an audit artifact. The corrected protocol is isolated under the
`fixedseed_v2` contract, model, prediction, sentinel, and result filenames;
`build_submission_5.py` accepts only that corrected contract.

`submissions/submission_5` is created only when the frozen joint model's pooled sealed-test
TSAUC is strictly greater than the same-split separate-model incumbent. It
contains no pretrained weights: Crunch `train()` must create `resources/` from
the provided labeled training iterator.
