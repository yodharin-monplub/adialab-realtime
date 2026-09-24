# Submission 16: pre-submission estimate

Prepared locally on 2026-09-06. Subsequently pushed at the user's request as
Crunch submission **10**, ID **78615**. No cloud run was started.
Plan for about **4 hours**, with a practical estimate of **3–5 hours**
of execution on `gcp-cpu`, excluding queue time. This is an extrapolation, not a
guaranteed runtime or a measured cloud execution of Model 16.

## Configuration

The prepared entry point sets `INFER_PARALLELISM = 12`. OpenMP, OpenBLAS, MKL,
NumExpr, LightGBM prediction, and XGBoost prediction use one thread per worker.
Four of the runtime's 16 vCPUs remain available to the runner and scoring.

Use `cloud_run_settings.json` when creating the run; it now includes the uploaded
`submissionId` 78615. It selects `gcp-cpu`, `forceFirstTrain: false`,
`trainFrequency: 0`, and `determinismCheck: true`. A Python variable alone cannot
set the server's force-first-training flag. `train()` is a resource-validation
fallback only; all fitted weights are bundled.

The competition runtime-definitions endpoint reported `gcp-cpu` as active,
16 vCPU / 64 GiB RAM, Python 3.12, `e2-standard-16`, and network forbidden.
The 16-GiB small-RAM alternative reported the same quota rate, so the 64-GiB
option provides useful memory headroom without a reported quota-rate penalty.
Source: [official Crunch API schema](https://api.hub.crunchdao.com/spec.json),
GET `/v1/competitions/structural-break-real-time/runtime-definitions`.

## Measurements and extrapolation

Both benchmarks used Python 3.12.14, the same installed inference requirements,
four spawned workers, and the same 24 development series: 13,528 online points.
Only features were read; no labels or holdout scores were used.

| Measurement | Submission 16 | Previous successful cloud model |
| --- | ---: | ---: |
| Local elapsed time, including worker startup | 74.116 s | 65.481 s |
| Sum of worker inference times | 263.820 s | 233.033 s |
| Largest observed worker RSS | 444.88 MiB | 441.11 MiB |

The previous model is the source of platform submission 9, recovered read-only
with hashes. Its successful cloud run 111908 took 29,542 seconds (8 h 12 m 22 s)
with four inference workers on 16 vCPU / 64 GiB, determinism checking enabled.
Its defensive training function validates resources rather than fitting.

The measured relative elapsed cost is `74.116 / 65.481 = 1.132`.
Scaling that historical cloud execution from four to twelve workers gives:

`29,542 seconds * 1.132 * (4 / 12) = about 3.10 hours`.

Allowing for imperfect worker scaling, scoring, installation and machine
variability supports a roughly four-hour planning estimate and 3–5-hour range.
The historical run already included determinism checking, so its cost is not
added a second time. Changes in hidden evaluation size can move runtime outside
this range; workload changes should approximately scale the inference portion.
Queue delay is excluded. The short local sample is not a full cloud-run test.

As a separate reference, extrapolating local throughput to the available
5,036,517 training online points gives approximately 2.55 hours at twelve
workers under ideal scaling, before platform overhead and cloud/local CPU
differences. Training-set size is only a proxy, not a claim about hidden data.

## Memory

The sampled maximum RSS is about 0.435 GiB per worker, or 5.21 GiB for twelve
independently loaded workers. Reserve roughly 12–16 GiB for workers, model-load
transients, and longer series, plus the platform's data and scoring memory.
The selected 64-GiB machine leaves substantial headroom. This is not a measured
whole-container cloud memory peak; the full runner and hidden data were not
part of the local timing process.

## Validation and artifacts

- Exported detector probabilities, feature-model probabilities, and complete
  final streaming predictions matched the independent original-model/batch
  reference exactly on 540 observations (maximum error zero).
- Repeated inference was bit-for-bit identical. The generator readiness and
  one-observation-at-a-time consumption contract passed.
- A defensive training test confirmed that no training data were consumed.
- The actual portable runtime processed all 13,528 sampled observations in an
  isolated Python 3.12 environment without sklearn, joblib, or training modules.
- Resource SHA-256 hashes are validated on load. The archive is checked for the
  configured API key and includes neither AGENTS.md nor training data.

Evidence: `submission_validation/parity.json`, `benchmark_w4_n24.json`,
`baseline_benchmark_w4_n24.json` (the latter two inside `submission_validation/`),
`submission_validation/package.json`, and
`upstream_cloud_submission9/RECOVERY_MANIFEST.json`.

Upload root: `submissions/submission16/`; entry: `main.py`; model directory:
`resources/`. Portable archive: `submissions/submission16.zip`.
The local name “Submission 16” does not set Crunch's next submission number.

The deployment is an all-five-fold bag of the saved Model 9 and Model 11
components. The exploratory research OOF score 0.629300 does not directly
evaluate this bag and is not a promised cloud score.
