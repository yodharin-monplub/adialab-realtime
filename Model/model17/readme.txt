Model 17: small fixed Model 16 / Model 12 / Model 6 diversity-blend study

Run from the repository root:
  .venv/bin/python -u models/model17/study.py --bootstrap 300
Tests:
  .venv/bin/python -m unittest discover -s tests -p test_model17.py -v

Eight predeclared convex blends in margin space; no parameter optimization,
new model fitting, cloud uploads, or cloud runs. Model 16 is the incumbent.
The study uses all 8,333 complete development series and retains original folds.
Protocol and input/source hashes are written before loading development targets.
Saved baseline and Model 12 fold scores must reproduce before screening.

Candidate gates, fixed before scoring: mean gain >= 0.00025, >= 4/5 fold wins,
positive paired whole-series bootstrap lower bound, and positive gain when the
recipe for each fold is selected using only the other four fold scores.
That last check is selection sensitivity, NOT fully nested validation: upstream
models and model development reused these folds. Bootstrap uncertainty is
conditional on saved predictions and omits adaptive model-selection uncertainty.

An accepted screen requires exact deployment-matched independent validation
before promotion. The reserved outer holdout was already revealed in Model 6;
it must NOT be described as fresh independent validation or reopened for tuning.
Model 16 remains deployed. No competitor test labels or outer targets are loaded.
See RESULTS.md and the source-hashed runs/ directory for outcomes and evidence.

Completed outcome:
  Best: 85% Model 16 + 15% Model 6, blended in logit/margin space.
  Local mean fold TSAUC: 0.629790217 versus Model 16's 0.629300070.
  Gain: +0.000490147; wins 4/5 folds.
  300-replicate whole-series bootstrap 95% gain interval:
    [+0.000049465, +0.000877630], conditional on saved predictions.
  The same 85/15 recipe is selected by all five leave-fold-out selectors.
  Development screening gates passed. Independent confirmation remains pending.
  No runtime bundle was built during the initial OOF screen.
  Subsequent preparation created submissions/submission17/ and submission17.zip.
  No cloud submission or run was created.
  Six new Model 17 tests and nine shared feature/metric tests passed.
  See VALIDATION_STATUS.md for the holdout limitation and next-step requirements.

Deployment preparation:
  See DEPLOYMENT.md and deployment_validation/ for parity, timing and CV-gap audit.
  12 workers, inference only, 16 vCPU / 64 GiB; about 4 hours estimated cloud time.
  The packaged Model 6 branch preserves the existing timeout_safe_v1 cloud
  profile, which differs from historical fold-specific OOF recipes.
  Subsequently uploaded at the user's request as platform submission 11,
  submission ID 78958, using the supplied project token. No cloud run started.
  See deployment_validation/push_receipt.json and remote_verification.json.
  cloud_run_settings.json now includes the uploaded submission ID.
