Model177 - the float32 lattice gate on Submission42 (2026-10-09, late-submission period)

CONTEXT
On 2026-10-09 the organisers reopened submissions ("late submissions", hidden leaderboard against the
public top 10, 5 submissions/day, 15 h per two weeks, never run on the out-of-sample data) and stated
that "the 100 locally available, and the 300 easily identifiable timeseries will remain in the
dataset". The user confirmed with the organisers that using the artefact is allowed.
The artefact (Discord #adialab-general, 2026-10-02, Farukcan): some series are real sensor readings on
a quantisation grid; the z-scoring keeps the grid, float32 storage keeps it for ~300 series across the
three sets, and the spliced synthetic break produces off-grid values right at the break.

WHAT IS IN THIS FOLDER
  lattice_gate.py   the detector + per-series gate that ships in the bundle (documented inside)
  scan177.py        scans a split with the gate exactly as the bundle drives it; writes scan_<split>.csv
  scan_train.csv    126 lattice series of the training set (step, step in ulps, trigger steps, tau)
  scan_test.csv     the 2 lattice series of the local test set (ids 10044 and 10079)
  validate177.py    runs submissions/submission42 and submission43 on the local test set and checks
                    bit-identical scores for non-lattice series, the firing step for lattice series,
                    and TS-AUC of both; writes validation_<N>.json
  build: submissions/submission43 = submissions/submission42 (copied, __pycache__ removed)
         + lattice_gate.py + new main.py (wraps solution147.infer) + readme.txt

HOW THE DETECTOR HAD TO BE BUILT (three failed attempts first)
  1. "unique-value ratio" - useless: the step is ~1e-5 so almost no exact duplicates occur.
  2. step = smallest gap, phase test over all values - found 2 series only: a single-gap step estimate
     is ~0.5% off (one float32 ulp over a 1e-5 step) and the phase is scrambled after ~100 lattice
     points; the series span ~1e5 points. Same reason the "lattice in first differences / drift"
     hypotheses looked plausible and were wrong.
  3. gating the step on the ulp of the LARGEST value rejected real lattices (one |v|=30 outlier).
  Final: progressive least-squares refinement of the step (gaps <= 20 quanta, then 400, 8e3, 1.6e5,
  4e6), joint base/step refit, step/{1..4} tried, per-value tolerance max(0.01, 2 ulp/step), values
  with tolerance > 0.3 ignored; accept when > 98% of unique history values are on the lattice and that
  beats the chance rate by 0.3. Candidate steps below 8 ulps are not testable and are skipped.

RESULTS - training set (scan177.py train)
  10,000 series -> 126 lattice series (1.26%): 64 with a break, 62 without; 6 histories already contain
  a few off-grid values ("unclean").
  rule first (fire on the first off-grid online value): exact 58/64, <=5 late 4, EARLY 2 (both unclean
    histories: 5643 fires 470 steps early, 6569 11 steps early), false alarms 1/62 (7253, one value)
  rule burst (2 of the last 3 decidable values off-grid): exact 0/64, all 64 within 5 (mean delay 1.2),
    no early, no false alarm
  rule auto (first for clean histories, burst for unclean): exact 57/64, <=5 late 7, early 0, never 0,
    false alarms 0/62, mean delay 0.12 steps                                  <- deployed
  Local test set: 100 series -> 2 lattice series, both with a break, both fire exactly at tau.
  Simulated TS-AUC gain (Other/tsauc.py, synthetic base scorer at 0.635 with the real labels, gate
  applied to the 126 series): +0.0086 (at a 0.595 base: +0.0099). The base model's own score on the
  lattice series does not matter: the gate replaces it.

VALIDATION (validate177.py 100 - the bundle run exactly as the platform drives it)
  100 local test series, 50,983 online steps: Submission42 TS-AUC 0.583984 -> Submission43 0.610763
  (+0.0268; the local set is 2% lattice series, both with breaks, so this overstates the gain).
  98 non-lattice series bit-identical to Submission42; 10044 fires at its break step 129, 10079 at 638;
  scores before the break <= 0.0096. Runtime +0.7% (859 s vs 853 s single process).
  Sealed by deployment/seal43.py: submissions/submission43.zip, 129,026,336 bytes, 177 files.

DEPLOYMENT
  submissions/submission43 (Submission42 + gate). Expected hidden-leaderboard score: Submission42's
  public score + ~0.008. See score.txt once scored.
  deployment/seal43.py -> push43.py --push (needs the user's word) -> start_cloud43.py ->
  python Model/run_score_any.py RUN_ID yodharin
  Reproduce: python Model/model177/scan177.py train; python Model/model177/scan177.py test;
             python Model/model177/validate177.py 100; zip the submissions/submission43 folder and
             upload it on the hub (same run options as Submission42: no training, INFER_PARALLELISM=12).
