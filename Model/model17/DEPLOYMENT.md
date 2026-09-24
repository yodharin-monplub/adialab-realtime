# Model 17 deployment preparation and CV/cloud gap audit

Prepared `submissions/submission17/` and `submissions/submission17.zip` locally.
Subsequently uploaded at the user's request as platform submission **11**,
ID **78958**. No cloud run has been started. Entry: `main.py`; model directory:
`resources/`. Local names are not Crunch submission numbers.

## Configuration and measured cost

The frozen deployment blends 85% of Model 16 and 15% of the previous Model 6
`timeout_safe_v1` cloud deployment in margin space. It shares the native
155-voter detector traversal, three feature-model bags, and detector features
once per observation. Both branches retain their original, different numerical
bagging conventions and all persistence states. State is per series, not global.

`INFER_PARALLELISM = 12`, with one numerical-library thread per worker.
`cloud_run_settings.json` selects `gcp-cpu` (16 vCPU / 64 GiB),
`forceFirstTrain=false`, `trainFrequency=0`, and `determinismCheck=true`.
The staged settings now contain uploaded submission ID `78958`.
All weights are bundled; `train()` only verifies hashes.

Matched Python 3.12.14 benchmarks used four independently spawned workers and
the same 24 development series, totaling 13,528 online observations. Both
history and online values were float32, matching Crunch's input transport.

| Runtime | Elapsed including startup | Maximum sampled worker RSS |
| --- | ---: | ---: |
| Model 16 | 66.738 s | 437.74 MiB |
| Model 17 | 72.849 s | 441.77 MiB |

Model 17's measured elapsed cost is 1.092 times Model 16's. The actual Model 16
cloud run 112662 took 13,021 seconds (3 h 37 m 1 s) with 12 inference workers
on the same planned runtime class, and determinism checking enabled.
Multiplying by the matched timing ratio gives **3.95 hours**. Plan for about
**4 hours**, with a practical **3.5–5.5-hour range**, excluding queue delay and
assuming comparable hidden workload. Scaling, setup, and scoring are not
guaranteed to track the short local benchmark proportionally.

Twelve times the sampled maximum worker RSS is about **5.18 GiB**. Reserve
12–16 GiB for workers, longer series and model-loading transients, plus platform
data/scoring memory. The 64-GiB runtime has substantial headroom, but the local
sample is not a measured whole-container cloud peak. The bundle is about
97.7 MiB uncompressed, only slightly larger than Model 16.

## Implementation checks

On 540 cloud-precision observations, the shared implementation matched both
independently running production models and the final blend exactly: maximum
error **zero** for each. Repeat inference was bitwise identical. Generator
readiness, one-point-at-a-time consumption, per-series frame release, and the
non-training fallback were checked. Only one expensive detector traversal and
one detector feature update occur per observation.

The cloud-like Python 3.12 environment runs without sklearn, joblib, or training
pipelines. Six blend-study tests and nine shared feature/metric tests also pass.

Evidence: `deployment_validation/parity.json`, `model16_w4_n24.json`,
`model17_w4_n24.json`, `audit.json`, and `package.json` (all in
`deployment_validation/`). Resource hashes are checked at load; the archive is
checked for the configured API key and excludes AGENTS.md and training data.

## Why cloud is below local CV

Model 16's local mean fold TSAUC is **0.629300070**. Its verified cloud TSAUC
is **0.621969456**: a difference of **0.007330614**, or **0.7331 percentage
points** on the percentage display. The previous Model 6 local/cloud gap was
about 0.0100, so Model 16 improved both results and narrowed this observed gap.

These scores do not evaluate identical models on identical data:

1. **Adaptive development reuse.** Detector selection, stack fitting, and model
   comparisons reused the development folds. The reported OOF score is useful
   for screening but is not an unbiased prediction of an external score.
2. **OOF versus production ensembles.** Local results combine fold-excluded
   predictions and fold-specific recipes. Deployment uses saved five-voter
   bags and all-development stack coefficients. Scoring these bags on the
   training series would be optimistic, not a replacement CV score.
3. **Different evaluation samples.** Sampling variation or distribution changes
   may contribute. The aggregate score gap alone cannot separate these causes.

We checked the concrete precision difference: Crunch transmits both history
and online values as float32, whereas previous local checks preserved history
as float64. Re-running the same Model 16 deployment on the 24-series sample
with full-precision versus cloud-precision inputs produced **identical output
predictions on all 13,528 points**. The sampled TSAUC change was zero. This
does not test every possible series, but gives no evidence that input rounding
explains the observed cloud/CV gap.

The precision audit's absolute training-sample TSAUC is deliberately not used
as validation: most saved voters trained on those development series. Only
the paired precision difference is informative for this diagnostic.

The transport behavior is documented in the
[official runner](https://github.com/crunchdao/competitions/blob/master/competitions/structural-break-real-time/scoring/runner.py).
Verified cloud run metadata is in `deployment_validation/model16_cloud_reference.json`.

## Additional Model 17 caveat

The OOF screening result **0.629790217** is not the score of this packaged
ensemble. Besides the five-voter/OOF distinction, the proven Model 6 cloud
profile excludes `core__feature__extra_trees__fisher_256__extra_deep` for runtime
safety. That component occurred in one of five historical held-fold recipes
and as a 5% last step in the original all-development recipe. The package
intentionally preserves the existing cloud profile rather than reinstating
the expensive model. This profile distinction is recorded in `ensemble17.json`.

Weights have not been retuned. Do not add the +0.000490 local screening gain
to the cloud score and call it a forecast. The reserved outer holdout was
previously revealed; it remains closed. Independent validation is still
pending, and Model 16 remains deployed until a future explicitly authorized
evaluation of this candidate.
