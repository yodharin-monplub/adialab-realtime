Model 21 — aligned splice weight follow-up to cloud-confirmed Model 20
===================================================================
Model20 cloud TS-AUC: 0.6234481054281362 (platform submission12, run114669).
This experiment changes ONLY synthetic training weight: aligned40/aligned60
versus the deployed aligned25 baseline. This is not a new data-generation
experiment: the same 1,000 donor pairs and 400,000 sampled synthetic rows per
fold are reused. Each fit also uses the same 400,000 real rows. Synthetic
rows therefore remain 50% of row count; weight is a separate quantity.
Nominal 40%/60% weight applies within covered (time-step,target) cells.
Uncovered cells retain their real weight, so actual total share is lower.

Continue the exact deployed XGBoost component: five folds, 256 original
features per fold, depth3, 800 trees, original seeds and hyperparameters.
Both donor series exclude validation and historical stopping folds. Original
Model20 cache blocks and frozen inputs are checksum-verified, opened read-only,
and reused without copying or overwriting. Model20 itself is unchanged.
First reproduce aligned25 fold0 predictions bit-for-bit as a runtime sanity
check, then run ten new fits. No additional recipes are selected mid-study.

Evaluate the FULL frozen ensemble on complete real development series, not
synthetic validation. Replace only this XGBoost component; all other members,
stack weights and online postprocessing stay fixed. Primary contrasts are
aligned40 vs Model20 and aligned60 vs Model20. Use 1,000 paired whole-series
bootstrap draws within folds and simultaneous 95% lower bounds for both.
Promotion requires gain >=0.0003, wins >=4/5 folds, and lower bound >0.
Report standalone component scores and no-break alarms as diagnostics.
Previously reused development folds are not independent confirmation;
bootstrap excludes upstream selection/refitting uncertainty. The reserved
outer holdout remains closed. No competition test labels are used.

Run/resume (CPU only, four XGBoost threads):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model21/run_model21.py run
Report/resume bootstrap after all fits:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model21/run_model21.py report
Run directory is recorded in models/model21/runs/latest.json.
Progress: per-fit checkpoints, preliminary_scores.json, bootstrap.npz,
then summary.json and RESULTS.md. Concurrent runs are prevented by flock.

No automatic packaging, upload, cloud run, or hibernation. Keep Model20 as
incumbent unless a candidate passes; a local pass still needs cloud validation.
