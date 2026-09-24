# Model73 / Submission27 deployment

The Model73 direct-voter variant was bundled separately as
`submissions/submission27` (SHA-256 of ZIP:
`dd8b92582980d3e36877815235527ce7088a2836088958203acad9ec3c985872`).
It was uploaded to the user-approved `sympathetic-trout` project as platform
submission number 2, ID 80912. Remote acceptance, validity and library checks
passed. The inference-only cloud run was **116613**, created with
`forceFirstTrain=false` and `trainFrequency=0`. It ended unsuccessfully on
2026-09-14 with exit code 124, exit reason `timeout`, after 35,572 seconds
(9h 52m 52s). The platform returned no prediction and therefore no TS-AUC
score. Available logs show startup and worker launch but no Python exception;
the platform's explicit timeout reason is the supported diagnosis.

Verification: 500 sampled held-fold LightGBM predictions matched frozen
checkpoints within 2.97e-8; 540 streaming predictions matched offline
reference outputs exactly on Python 3.14 and cloud Python 3.12. Inference was
deterministic, incremental and did not consume training data. The 24-series,
two-worker benchmark used 13,528 online observations: Model26 baseline 152.59
seconds wall/291.13 worker seconds; Model73 147.74 seconds wall/280.62 worker
seconds; Model73 peak worker RSS 413.09 MiB. Relative to Model26's actual
20,237-second cloud run, the nominal estimate is 5.62 hours (planning range
4.5–7.3 hours). Twelve measured workers extrapolate to 4.84 GiB RSS, excluding
parent/platform memory. This timing estimate was materially optimistic on the
cloud workload; the matched 24-series sample did not predict the full run.

Exploratory reused-development OOF is 0.6354292998; it is not the cloud score
and not the full Model62 research recipe. The frozen outer holdout and test
labels remain sealed. The score monitor is paused and no retry has been started.
