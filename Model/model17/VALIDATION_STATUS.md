# Model 17: promising screen, independent confirmation pending

The eight-recipe study selected **85% Model 16 + 15% Model 6** in margin space.
Mean development-fold TSAUC is **0.629790217**, versus **0.629300070** for Model 16.
The +0.000490147 gain is positive in four of five folds. Each selector trained
on four fold scores independently chose the same 85/15 recipe for the fifth.
The paired whole-series bootstrap interval is [+0.000049465, +0.000877630].
All four predeclared development screening gates passed.

This is still conditional, reused-development evidence. The bootstrap interval
does not account for choosing the winner or the upstream model-selection history.
It is not a forecast of the cloud score. Do not add the local gain to the cloud
score and present the sum as a predicted result.

## Correction to the proposed validation plan

The reserved outer split is **not untouched**. The existing repository explicitly
records prior exposure in:

- `models/model9/README.md`: the outer holdout was previously revealed and this
  package does not predict or score it.
- `models/model7/README.md`: the revealed Model 6 holdout is prohibited from
  Model 7 modeling steps.

The study did not load those labels or predictions. Reopening that split would
not provide the genuinely independent confirmation originally proposed. A new
random split of the same data would not magically remove upstream exposure
either. Fully nested base refits can improve the validation design, but must
still be distinguished from genuinely new external observations.

## What remains before promotion

1. Keep the 85/15 weights frozen; do not expand the grid after observing this win.
2. Implement an exact streaming blend of the saved five-voter Model 16 and
   Model 6 deployments. Their expensive primitives overlap, but their precise
   detector bagging/rounding conventions differ; do not replace one with the
   other without parity checks. This implementation has now been prepared and
   benchmarked; see DEPLOYMENT.md for the production-profile distinction.
3. Validate stream parity, generator causality, and per-worker timing/memory.
4. Obtain independent evidence on genuinely fresh permitted data, or explicitly
   authorize one frozen cloud evaluation while treating leaderboard feedback
   as limited exploratory evidence rather than a fresh validation set.

No submission or cloud run was created. Model 16 remains unchanged. The six
new blend/selection/gate tests and nine existing feature/metric tests passed.

Subsequent deployment preparation: submissions/submission17/ now contains the
shared-inference runtime. It matches both production branches and the final
blend exactly on 540 observations. This is implementation parity, not new
independent score evidence. The original Model 6 OOF recipe differs from its
timeout-safe cloud profile, as documented in DEPLOYMENT.md.
