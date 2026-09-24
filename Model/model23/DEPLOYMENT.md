# Submission 23 — uploaded as platform submission 13

## Verified cloud result (2026-09-11 UTC)

Run 114854 completed successfully, prediction 93349, submission ID 80033.
TS-AUC: **0.6237894019876368** (API display 62.37894019876368,
verified metric multiplier 100). Gain versus Model20: **+0.0003412965595006**.
Model23 is now the cloud-confirmed incumbent. Duration 14,693 seconds
(4h 04m 53s), forceFirstTrain=false. Model24 and Model25 both failed local
promotion gates and are not submitted.

Upload accepted and read back successfully: submission number 13, ID 80033.
Platform valid=true and librariesVerified=true. See
`deployment_validation/push_receipt.json` for the authoritative receipt.
No cloud run was started by the upload. Saved run settings now reference ID80033.

Selected recipe: equal arithmetic probability mixture of Model20 aligned25 and
Model21 aligned60 **within each fold**, followed by the existing five-fold
logit bag. All original Model20 resources remain unchanged; five paired native
XGBoost files are added. This is not a flat ten-voter logit average.

Local full-ensemble CV: 0.6320947258732259 versus 0.6315755373999716 for Model20.
Gain +0.0005191884732542506; 4/5 fold wins; simultaneous lower95%
+0.00009188372837007905. All 1,000 paired bootstrap draws were verified.
This is reused-development evidence, not independent or cloud confirmation.

## Verification

- 59 regression tests passed.
- 540 streaming observations: zero detector, component, and final prediction error.
- Repeated inference bitwise identical; incremental input consumption verified.
- The train entry point validates resources without consuming training data.
- Python3.12.14 smoke test: zero prediction error on the same 540 observations.
- All original model-resource hashes preserved; exactly five model files added.
- Archive contents, checksums, CRCs, upload file inventory, and API-key exclusion checked.

## Cloud settings and estimate

Inference only: forceFirstTrain=false, trainFrequency=0, 12 inference workers.
Target16vCPU/64GiB. Each numerical library uses one thread per worker.

Matched benchmark: 24 real development series, 13,528 online observations,
two workers, Python3.12, sequential baseline/candidate runs. A separate warm-up
was excluded. Model22 training continued during both matched measurements.

Model20:145.526949s wall,274.968282s summed worker time,381.441MiB peak worker RSS.
Model23:155.121764s wall,294.701894s summed worker time,387.879MiB peak worker RSS.
Conservative time ratio:1.071766867, the maximum of wall ratio, summed-worker
ratio and1.0. Reference Model20 cloud run114669 completed in13,750s, verified
through the read-only Crunch API.

Nominal Model23 cloud estimate: **4h06m** (4.093554h); planning range3.3–5.3h,
excluding queue time and subject to host/platform overhead variability.
Twelve times measured worker RSS is4.545GiB, excluding parent/platform overhead.

## Artifacts

- Bundle: `submissions/submission23/`
- Archive: `submissions/submission23.zip` (73,647,547 bytes)
- Archive SHA256: `bc7d674b74402aa5210a929b8561305a7986b8d346a0fbbf6c0ae5e92b5c5956`
- Upload inventory:70 code files,21 model files.
- Evidence: `models/model23/deployment_validation/`
- Build/validate/seal scripts: `models/model23/deployment/`

The submission was uploaded on user direction; no cloud run was started.
Model22, Model24 and Model25 completed without passing their promotion gates.
