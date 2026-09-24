Model22 — retain all real fitting data and append aligned synthetic data
======================================================================
User-approved follow-up to Model20 (cloud TS-AUC 0.6234481054). Model21's
higher synthetic weight did not pass validation; keep the real data intact.

Two fixed training arms, five folds each:
  all_real: every eligible real training row; no 20% screen or 400k cap.
  all_real_plus_synthetic: identical real rows and weights plus the existing
    400,000 aligned synthetic rows from 1,000 fitting-only donor pairs/fold.
All real means all rows in the three allowed fitting folds; validation and
historical stopping folds remain excluded, along with the reserved holdout.
Do not train on validation simply to enlarge the data set.

Compute the original opposite-class/time-step pair weights on the complete
real fitting set, with the same mean-positive-weight normalization. Freeze
these real weights for BOTH arms. Synthetic rows inherit the weight of a
real row in the same (time-step,target) cell; unsupported cells get zero.
No real weights are reduced, no fixed synthetic weight percentage is imposed,
and the combined weights are NOT renormalized. Record actual synthetic share
and real-row/weight checksums. Adding weight also changes effective fixed
regularization; this is part of the append-data intervention, not isolated
evidence of new information alone. Synthetic labels remain assumptions.

Retain original XGBoost features, depth3/800 trees, all parameters and seeds,
including per-tree stochastic subsample .85. Thus every real row is available
to fitting, but individual trees retain the original sampling behavior.
Keep the full ensemble and postprocessing fixed except this XGBoost component.
All existing Model20 caches are verified and opened read-only. Matrix filling
is blockwise to avoid allocating the larger 439-column union-feature copy.
Four CPU threads, one fit at a time. No GPU; no new synthetic feature cache.

Validation uses only complete REAL held-fold series. Reconstruct Model20 as
the incumbent baseline. Three fixed contrasts: all-real vs Model20, augmented
vs Model20, augmented vs all-real. Run 1,000 paired whole-series bootstrap
draws within folds; simultaneous95% bounds cover all three contrasts.
All-real promotion: >=.0003 gain, >=4/5 wins, lower bound>0 vs Model20.
Augmented promotion: same gates against BOTH Model20 and all-real control.
Report component scores and no-break alarms as diagnostics. Do not mistake
reused-development gains for independent or cloud confirmation. The previously
revealed outer holdout remains closed. No competition test labels are used.

Run/resume:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model22/run_model22.py run
Report/resume bootstrap:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model22/run_model22.py report
Progress: runs/latest.json, per-fit .fit.json/.npz/.ubj checkpoints,
preliminary_scores.json, bootstrap.npz, then summary.json and RESULTS.md.
All sources, data, donor plans and settings are frozen before training.
No automatic packaging, submission, cloud run, or hibernation.
