# Frozen feature-selection contract

This is an experimental successor to `submission_2`. It contains no pretrained
model. Crunch's `train()` fits both LightGBMs from its labeled training iterator
and writes every learned weight under `resources/`.

The two small `*_feature_selection.npz` files contain only integer source-column
IDs and matching feature names. They freeze a selection derived exclusively
from the local `train/` data before the locked test was scored.

## Leakage-safe protocol

- train: whole-series folds 2-4 (4,800 series);
- validation: whole-series fold 1 (1,600 series);
- locked test: whole-series fold 0 (1,600 series);
- rankings used train rows only;
- subset methods and blend weights were chosen on validation only;
- the locked test was evaluated once after the contract was frozen.

Methods tested separately for core and candidate were all features; LightGBM
gain, split count, mean absolute SHAP, and consensus top 64/128/256/512/768;
nonzero gain; gain plus 0.98 absolute-correlation pruning; cumulative feature
banks; and a random-256 negative control excluded from deployment selection.

## Frozen configuration

- core: consensus top 768, 101 rounds, logit weight 0.55;
- candidate: gain ranking plus correlation pruning, 671 columns, 217 rounds,
  logit weight 0.45;
- contract SHA-256: `e7154880585cb472bbed94079048312641d5b4f03d6dfc912ac4a8e908861a8f`.

Validation TSAUC improved from 0.614042 to 0.621166. Locked-test TSAUC moved
from 0.630459 to 0.630597 (+0.000138). The paired five-subfold delta had SE
0.001653 and 95% t interval [-0.004187, 0.004994], so the improvement is not
statistically resolved. Treat this as a lower-dimensional cloud experiment,
not as a demonstrated replacement for submission 2.
