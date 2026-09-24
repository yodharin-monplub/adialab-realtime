Model108 / Submission29: public-test overlap diagnostic
=====================================================
Status: PREPARED LOCALLY. No upload, submission, or cloud run performed.

Purpose
Check whether the 100 official X_test.reduced.parquet series occur among the
series supplied to cloud inference. The actual cloud overlap is still unknown.
The original public input parquet is included in resources (100 series,
346019 rows). No test labels are needed or included.

Entry point: main.py; model directory: resources; INFER_PARALLELISM=1.
CPU only. Suggested future run settings: forceFirstTrain=false,
trainFrequency=0. train() only validates the included reference data.
This is a diagnostic submission. Every online prediction is the constant 0.5;
its leaderboard score should not be used to assess the existing trained models.

Matching
Compare SHA-256 fingerprints of the complete ordered history and then the
complete online stream, checking both lengths and the history/online boundary.
A shared history alone is not a full YES. Cloud IDs and time labels are not
used to establish a match. The online iterator is consumed only one value per
prediction; matching never reads future cloud values in advance.

Two matching modes are reported separately:
- exact: exact numeric equality after converting values to float64 and treating
  signed zeros equally. This is the primary YES percentage.
- public_float32_precision: equality after converting to the public parquet's
  float32 precision. This detects copies served with tiny float64 differences;
  it does not establish bit-for-bit float64 equality.
Reordered, rescaled, shortened, or differently split sequences are not full
matches. Nonfinite inputs fail explicitly. Hash collisions are negligible.

Final output
OVERLAP_FINAL is one JSON line, followed by a readable percentage summary.
For each matching mode it includes:
- yes_cloud_series / completed_cloud_series * 100 = yes_percentage.
- reference_series_found / 100 * 100 = reference_found_percentage.
- history_only_yes_cloud_series and history_reference_series_found as
  separate history-match counts (including histories of full matches).
Repeated cloud copies count as separate cloud YES series, but each public
reference ID is counted only once in reference coverage. Identical public
references, if any, are indistinguishable by values and all such IDs count.

Example ONLY: if all 100 references appear exactly once in 10000 cloud series,
the final line says YES = 100/10000 (1.0000%); public test found = 100/100
(100.00%). This example is not a cloud result.

The denominator is the actual number of completed series, never a hard-coded
10000. expected_cloud_count_reached flags whether 10000 were observed. Counts
reset for each infer invocation; INFER_PARALLELISM=1 prevents worker-local
counter splitting when the runner respects that setting. Separate phases,
determinism replays, or externally sharded invocations produce separate reports;
do not add their percentages. A report with fewer than 10000 series cannot
establish absence from all 10000. complete means that invocation exhausted its
input, not that the platform necessarily supplied its whole hidden dataset.
A closed or failing generator prints a PARTIAL report for completed series;
a process killed by the platform may not execute finalization. Progress is
printed every 1000 completed series. No hidden values or IDs are printed.

Build and verify from project root:
  .venv/bin/python models/model108/build.py
  .venv/bin/python models/model108/validate.py
Prepared artifact: submissions/submission29.zip
Upload root, when separately requested: submissions/submission29
The build uses an explicit file inventory and verifies every archive hash.
Both commands are local and never submit. The reference parquet was downloaded
from the authenticated official public submission-phase data release; its
size and hash are recorded in reference_source.json. No credentials are bundled.
