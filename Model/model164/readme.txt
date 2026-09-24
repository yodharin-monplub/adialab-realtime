Model164 - dev-fold twin of the Submission38 recipe, and the blend weight
(2026-09-23).

run_model164.py = Model158's student (oracle-teacher soft targets on real
rows, splices and re-cuts) with the THREE re-cut sets pooled (Model163),
trained as 3-fold voters so it has out-of-fold predictions (arm SRP40).
  dev vs SRB40: +0.0039 -0.0004 +0.0032 +0.0032 +0.0047 (mean +0.0029, 4/5)
  vs the Submission33-era student SRE40: about +0.010 per fold.

BLEND WEIGHT (weights_eval164.py = Model148's protocol with this student:
weights picked on pilot folds 0-1, confirmed on folds 2-4)
  reference ws=0.5 wg=0.5 -> dev mean 0.64149
  ws=1.0 wg=0.5: pilot +0.00094, confirm +0.00055, 5/5
  ws=1.5 wg=0.5: pilot +0.00111, confirm +0.00034, 3/3 confirmation wins
  -> pilot-selected ws=1.5, wg stays 0.5.
SEALED HOLDOUT with the dev-only voters (one check, weights chosen on dev):
  ws 0.5 -> 0.635732 | ws 1.0 -> 0.636847 | ws 1.5 -> 0.637246
  (Submission33: 0.634478) -> +0.0028 at ws=1.5.
The student had outgrown its weight: it is now stronger than the chain on
the holdout (student alone 0.6352 vs chain 0.6297).

SUBMISSION39 = Submission38's all-series voters (arm AP1) with the student
weight 1.5 in the bundle manifest (build35.py WSTUDENT=1.5; the serving code
reads that field). validate39.py rebuilds the expected predictions at the new
weight: parity 1.1e-7, replay 0.0, ~5 cloud hours; sealed
(submissions/submission39.zip, 130 MB). Emailed for approval.

BUG (found 2026-09-23, after Submissions 38/39 were sealed): in
models/model154/run_all.py the oracle-teacher flag was
`arm.startswith(("AB", "AS"))`, so the pooled "AP" arms used the OLD 4-fold
teachers for the REAL rows (the synthetic rows were fine). Submissions 38/39
are therefore weaker than the measured recipe and were NOT uploaded.
Fixed to `("AB", "AS", "AP")`; arm AP3 (SYNA=0.3) retrained with it ->
SUBMISSION40 (WSTUDENT=1.5): parity 1.1e-7, replay 0.0, ~6 cloud hours,
submissions/submission40.zip. This is the bundle that matches the holdout
measurement 0.64038 (dev-only twin) vs Submission33 0.63448.

CLOUD RESULT: Submission40 (minimum-mantis run 119771, 4.7 h) scored
public TS-AUC 0.63628, rank 80 (Sub37 0.63403 rank 93, Sub33 0.63311).
Holdout twin predicted ~0.6404, so the holdout->cloud offset is about
-0.004 here (it was -0.0014 .. -0.0022 before; the recipe changed a lot).
