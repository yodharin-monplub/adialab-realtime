Model23 — equal-weight mixtures of existing synthetic-weight voters
=================================================================
Lightweight independent study while Model22's full-data fits run. No new
model fitting, detector features, or synthetic series generation. Do not
change Model22, its frozen protocol, or its selection rules.

Two predeclared mixtures for the core XGBoost component:
  mix_40: 50% Model20 aligned25 probability + 50% Model21 aligned40 probability.
  mix_60: 50% Model20 aligned25 probability + 50% Model21 aligned60 probability.
These are probability mixtures BEFORE the existing component logit/linear
stack and causal rolling postprocessing. All other members/weights stay fixed.
No blend-weight search or selection based on partial Model22 results.
Unblended Model21 candidates slightly improved mean local ensemble CV but
failed confidence gates. Averaging may reduce variance; it may also fail.

Use held-fold voters only, with the same original fitting/validation roles
and donor exclusions. Source checkpoints and model artifacts are verified.
The untouched Model20 ensemble is the current cloud-confirmed incumbent.
Compare both mixtures against it on all five complete real validation folds,
using 1,000 paired whole-series bootstrap draws and simultaneous95% bounds.
Pass: mean gain>=0.0003, wins>=4/5 folds, simultaneous lower bound>0.
All folds have been reused in prior experiments; this is exploratory evidence,
not independent confirmation. Bootstrap does not account for upstream model
fitting or research selection. No reserved holdout or competition test labels.

CPU-only evaluation, one numerical thread. No automatic packaging or upload.
A future deployed mixture would require ten voters for this component,
serving parity checks and a new inference benchmark before submission.

Run/resume:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model23/run_model23.py run
Report/resume bootstrap:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model23/run_model23.py report
Results: runs/latest.json, preliminary_scores.json, bootstrap.npz,
summary.json and RESULTS.md. Model20 and Model22 remain unchanged.
