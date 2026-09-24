Model 20 — retrain a strong deployed backbone with synthetic splice data
=====================================================================
This is NOT another compact standalone model. It retrains the exact
core:feature:xgboost__fisher_256__xgb_depth3 component used by the Model9
stack inside the approximately0.622-cloud-score Model16 submission.

Run/resume from repository root (Python3.14 training environment):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model20/run_model20.py run
  .venv/bin/python models/model20/run_model20.py report
CPU only. Four feature processes, four XGBoost threads; no cloud activity.

Fixed fifteen-fit study: five folds x control/raw10/aligned25. These two
splice recipes were chosen from prior experiments before this experiment's
results. Each fold keeps its ORIGINAL 256 Fisher-selected detector features,
original XGBoost parameters, 800 trees, model seed, 20% deterministic row
screen and uniform cap of400,000 real rows. Hyperparameters include depth3,
learning_rate .05, min_child_weight10, subsample .85, colsample .65,
reg_alpha .2, reg_lambda6, max_bin127. Original XGBoost had no early stopping;
we preserve its three-fold training role and still exclude the separate
historical stopping fold from both real fitting and synthetic donors.

Both donors must belong to fitting folds. Up to1,000 unique A donors/fold;
>=32 online prebreak and remaining postbreak points, distinct B, reuse<=5.
B contributes a contiguous true postbreak segment with no padding/repetition.
A's history, prebreak prefix and total horizon are preserved. Raw B remains
unchanged; aligned B receives the difference of donor prebreak medians over
last up to128 clean observations. Composite inputs float32; source real
inputs retain historical float64 precision. Only fitting composites receive
constructed labels. Synthetic labels are assumptions, not independent truth.
Seed20260920+fold; row substream SeedSequence([seed,1]); <=400k synthetic rows.
Both arms share donor pairs and row samples. Real weights use FULL original
fitting-fold (step,target) counts, matching the original backbone training.
Move10% or25% of each covered cell's sampled-real weight to synthetic rows;
uncovered cells keep all real mass. No global weight renormalization.

Rebuild just439 union feature columns from18 deployed detector banks; all
causal prefix features are computed before sampling rows. Validate original
native XGBoost prediction parity on15 real development series first. Real
cache ~7.4GB, synthetic caches <=4.1GB, plus metadata/models. Caches commit
checksummed32-series blocks and resume from their last committed prefix.
All sources/inputs/dependencies/donor assignments are frozen before fitting.
The native feature-parity check is separate from fresh refit reproducibility:
new dependency versions can change a fresh fit, so report its drift and compare
augmentation against BOTH the fresh paired control and original Model16.

Full ensemble evaluation: replace only the XGBoost column in each frozen
Model9 fold's linear stack using its saved coefficient. Recompute Model16's
causal rolling correction; keep compact Model11 and all other weights fixed.
Identity replacement must reproduce original Model16 margins bit for bit.
This is a fixed-stack ablation, not a newly fitted stack or fully nested CV.
If this helps reliably, refitting other components/restacking is a separate
next experiment; no automatic full rebuild or submission occurs here.

Score complete, unmodified real validation series. Compare standalone
component and complete ensemble. Whole-series paired bootstrap:1,000 draws
within original folds, seed200906. Simultaneous95% lower bounds cover four
full-ensemble contrasts (two recipes vs fresh control and original incumbent).
Pass requires gain>=.0003, wins>=4/5, lower bound>0 against BOTH baselines.
Fixed .5/.8 sigmoid-margin false alarms are diagnostics, not calibrated risk.

All development folds have been repeatedly used; existing feature selection
and stacking are development-informed and not fully nested. Bootstrap is
conditional on predictions and excludes prior research/refitting uncertainty.
The historically revealed outer holdout remains closed. No competition test
labels, uploads, deletion or modification of original model/submission files.

Subsequent user-approved deployment preparation:
  Submission20 replaces the five deployed XGBoost voters with aligned25 fits.
  Bundle: submissions/submission20/; sealed archive: submissions/submission20.zip.
  Estimated cloud runtime:3h48m; planning range3-5h excluding queue time.
  forceFirstTrain=false; trainFrequency=0; inference workers12, target16CPU/64GiB.
  See DEPLOYMENT.md and deployment_validation/push_receipt.json for upload status.
  No cloud run is started by packaging or upload. Original Submission16 preserved.
  Upload accepted: Crunch platform submission12, ID79379, minimum-mantis.
  Remote acceptance, platform validity and library verification passed.
