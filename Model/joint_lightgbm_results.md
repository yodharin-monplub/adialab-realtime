# Joint LightGBM experiment results

## Leakage contract

- Fresh series-grouped split seed: `20260821`.
- Train/validation/test: 60%/20%/20% complete series.
- Feature ranking used train only; subset and parameters used validation only.
- The sealed test was scored once after `9c1922738e7b9fd63a318eefd749feecc1732b0c29ddf8c9626aa390a1583e03` was frozen.
- The detector bank is fixed and every feature has offline/streaming parity.
- Historical experimentation means this is a procedurally fresh holdout, not a
  claim that these training series were never seen during earlier research.

## Selected joint model

- Selection: `balanced_top_768`
- Features: `768` total: `304` core, `387` candidate, and `77` from the new
  winner-inspired bank.
- Selected new-feature mix: 28 robust scale/location, 22 expected-shortfall,
  14 in-band/run, 7 db4, 5 distribution-distance, and 1 Hurst column.
- LightGBM tuning configuration: `leaves95`
- Validation TSAUC: `0.616458`
- Sealed-test TSAUC: `0.614092`
- Five-series-group mean ± SE: `0.613468 ± 0.021163`
- Complete-series bootstrap SE: `0.010389`
- Bootstrap 95% interval: `[0.593701, 0.632694]`

## Same-split incumbent

- Separate selected core/candidate LightGBMs.
- Validation TSAUC: `0.615903`
- Sealed-test TSAUC: `0.615405`
- Five-series-group mean ± SE: `0.614219 ± 0.017257`
- Joint minus incumbent pooled TSAUC: `-0.001313`
- Joint minus incumbent subfold mean: `-0.000751`
- Paired SE: `0.005296`
- Paired 95% t interval: `[-0.015454, +0.013952]`
- Paired complete-series bootstrap SE: `0.005772`
- Paired bootstrap 95% interval: `[-0.012181, +0.009776]`

## Submission decision

`submission_5` eligibility: **FAIL**.
The folder is built only when the frozen joint model has strictly higher pooled
sealed-test TSAUC than the same-split incumbent. No post-test retuning is allowed.
