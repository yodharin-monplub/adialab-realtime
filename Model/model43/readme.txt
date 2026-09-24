Model43: depth-3/4/5 diversity ensemble.

Model26 combines a depth-3 synthetic-data voter pair with a depth-4 pair and
improved both local OOF and the cloud score. Model43 trains the previously
untested depth-5 aligned-25 and aligned-60 voters while holding folds, rows,
synthetic examples, weights, feature subsets, rounds, and all other XGBoost
parameters fixed.

The primary probability is the equal-weight mean of the depth-3, depth-4, and
depth-5 voter pairs. It then enters the frozen stack-column replacement and
receives Model38's independently confirmed causal smoother. Raw equal-depth
output is retained as a diagnostic.

CPU only, four XGBoost threads. No packaging, upload, or cloud submission.

