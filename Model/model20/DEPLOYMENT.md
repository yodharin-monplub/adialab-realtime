# Submission 20

Uploaded successfully as **Crunch platform submission 12**, submission ID
**79379**, for `minimum-mantis`. Remote acceptance, platform validity and
library verification all passed. No cloud run was started.

Prepared from the original sealed Submission 16. Only its five core XGBoost
voters are replaced with Model 20's `aligned_25` models. All other model
resources, feature subsets, stack weights and blend settings are unchanged.

The local study improved mean-fold TSAUC from 0.629300070 to 0.631575537,
with five fold wins and a simultaneous 95% lower gain of +0.001437625.
This is reused-development evidence, not a new cloud score. The production
five-voter bag follows Model 16's deployment design; OOF results evaluate
individual held-fold voters, not the production bag.

## Validation

- Forty regression tests passed.
- Independent native-model/batch versus streaming parity: zero error on 540 observations.
- Repeated predictions bitwise identical; one-observation-at-a-time consumption passed.
- The cloud `train` function does not consume training data; it validates resources only.
- Python 3.12 cloud-compatible environment: zero output difference on 540 observations.
- Source Submission 16's 84 sealed file hashes remain unchanged.
- Archive and resource hashes verified; credentials excluded from the package.

## Cloud configuration and estimate

`forceFirstTrain=false`, `trainFrequency=0`, `INFER_PARALLELISM=12`.
Target machine: 16 vCPU / 64 GiB (`gcp-cpu`).

On the same 24 series / 13,528 observations with four local workers,
Submission 16 took 68.09 seconds and Submission 20 took 71.53 seconds.
Scaling the prior actual 13,021-second Model 16 cloud run gives about
**3 hours 48 minutes**, with a planning range of **3–5 hours**, excluding queue
time. Peak measured worker RSS was 378.47 MiB; twelve such workers total
approximately 4.44 GiB before platform/parent overhead. This is an estimate,
not a guarantee of cloud runtime or total memory usage.

## Files and commands

- Bundle: `submissions/submission20/`
- Sealed archive: `submissions/submission20.zip`
- Validation: `models/model20/deployment_validation/`
- Upload status and platform number: `deployment_validation/push_receipt.json`
- Cloud-run configuration: `cloud_run_settings.json`

The report's NumPy-boolean serialization error was repaired without changing
training, predictions, or bootstrap draws. `summary.json` records the original
and repaired report hashes. The saved results can be regenerated with:

```bash
.venv/bin/python models/model20/run_model20.py report
```

Uploading does not start a cloud run. The push tool refuses to upload again
when an acceptance receipt already exists. Check the receipt before retrying
an interrupted upload.
