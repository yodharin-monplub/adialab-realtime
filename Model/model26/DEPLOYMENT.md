# Model26 deployment status

The frozen research run `3169c9799a99a452` completed successfully.
Independent checkpoint/hash and promotion-gate verification passed:

- Full ensemble CV: 0.6324069786337266.
- Model23 reference CV: 0.6320947258732259.
- Gain: 0.0003122527605007; four of five fold wins.
- 1,000 paired whole-series bootstrap draws; lower 95% bound: 0.0001060272329128.

These are reused development scores, not a cloud result. Model23 remains
the cloud-confirmed incumbent at TS-AUC 0.6237894019876368.

Separate unsealed bundle: `submissions/submission26`.
The bundle preserves Model23's original model resources and adds ten depth4
XGBoost voters. Nested float32 probability averaging occurs within each fold,
before the unchanged five-fold logit bag and downstream stack/postprocess.

Verified serving evidence:

- 540 streaming observations: zero detector, feature, and final-output errors.
- Repeated inference is bitwise identical.
- Online observations are consumed incrementally.
- The training entry point does not consume training data.
- Cloud Python 3.12 compatibility: 540 observations, zero output error.
- Native nested-averaging unit test passed, including extreme probabilities.

The two-worker, 24-series candidate benchmark completed: 13,528 observations
in 147.38 seconds, with 281.65 summed worker seconds and peak worker RSS
440.31 MiB. Twelve times this measured RSS is approximately 5.16 GiB, excluding
parent process and platform overhead. A separate matched baseline benchmark
completed in 115.79 seconds (220.65 summed worker seconds). The conservative
candidate/baseline ratio is 1.2764. Applied to Model20's verified 13,750-second
cloud run, estimated Model26 runtime is 4.875 hours (about 4 hours 53 minutes),
with a planning range of 3.9–6.3 hours excluding queue time.
The candidate benchmark initially overlapped the short cloud smoke test;
account for this and a brief four-second unit-test overlap as possible
conservative timing biases when interpreting results. All 70 unit tests passed.
Model26 was uploaded with the new .env credential to the user-approved
sympathetic-trout project (user 4770): platform submission 1, ID 80134.
Remote acceptance, platform validity, and library verification passed.
No cloud run was started by the upload script.

The validated package is sealed as `submissions/submission26.zip`:
77,036,785 bytes, 70 code files, 31 model files.
SHA-256: `2e04c6fffbedeb0f99ec422b113c8f528fe7d3e9c3165d093efc91d4861fc683`.
Archive hashes and credential exclusion checks passed. User explicitly approved
uploading Model26 to sympathetic-trout. Receipt: deployment_validation/push_receipt.json.

Planned cloud configuration: forceFirstTrain=false, trainFrequency=0,
12 inference workers, one numerical thread per worker, 16 CPU / 64 GiB RAM.
