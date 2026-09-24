"""Fixed CPU experiment recipes, declared before scoring."""
CONFIGS = {
    11: dict(name="multiscale_gbdt", kind="lightgbm", fit_rows=500_000, rounds=350,
             parameters=dict(num_leaves=15, learning_rate=.04, min_data_in_leaf=150, lambda_l2=20)),
    12: dict(name="multiscale_extra_trees", kind="extra_trees", fit_rows=220_000,
             parameters=dict(n_estimators=160, max_depth=16, min_samples_leaf=60, max_features=.7)),
    13: dict(name="multiscale_pairwise", kind="pairwise", fit_rows=400_000, rounds=350,
             parameters=dict(num_leaves=15, learning_rate=.04, min_data_in_leaf=150, lambda_l2=20), pairs=500_000),
}
SEED = 20260906
BLEND_WEIGHTS = (.1, .25, .5)
