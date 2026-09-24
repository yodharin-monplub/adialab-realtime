# Push Submission 6

This directory is the complete submission root. It can be initialized without
downloading competition data, an older submission, or an older model:

```bash
cd /home/hp/resources/structural-break-real-time-minimum-mantis/submissions/submission_6/submission_ready
read -rsp "Crunch clone token: " CRUNCH_CLONE_TOKEN
echo
crunch setup-notebook structural-break-real-time "$CRUNCH_CLONE_TOKEN" \
  --submission scratch \
  --no-data \
  --no-model
unset CRUNCH_CLONE_TOKEN
chmod 600 .crunchdao/token
```

The generated `.crunchdao/`, `data/`, and `prediction/` paths are local CLI
state and are excluded from the upload. Push from this exact directory:

```bash
cd /home/hp/resources/structural-break-real-time-minimum-mantis/submissions/submission_6/submission_ready
crunch push --no-pip-freeze -m "submission 6: timeout-safe model6 ensemble"
```

Alternatively, upload `submission_ready/` through Crunch's file-selection
interface. Do not upload its parent `submission_6/` directory.

Before pushing, the directory must contain `main.py`, `solution.py`,
`feature_pipeline.py`, `detector/`, and the populated `resources/` directory.
Do not delete the resources: this submission deploys the exact five-fold
frozen Model 6 voters and `train()` validates them instead of retraining.

Useful checks from the repository root:

```bash
.venv/bin/python -m compileall -q \
  submissions/submission_6/submission_ready
.venv/bin/python \
  submissions/submission_6/submission_ready/validate_bundle.py \
  --steps 5
cd submissions/submission_6/submission_ready
crunch test --no-determinism-check
```

`crunch test` requires local competition data. The completed release check
scored all 50,983 reduced-test points in 4 minutes 35 seconds with four workers.
The lightweight validator also verifies force-first-train behavior and
deterministic one-pass streaming without consuming test labels.

`resources/manifest.json` records every model path and SHA-256 hash.
`bundle_manifest.json` records every file shipped in this directory. The
expected sealed-holdout TSAUC is `0.626144` with bootstrap SE `0.011365`.
