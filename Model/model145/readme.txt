Model145 - external data: the 2025 ADIA Lab Structural Break Challenge series
(AGENTS.md 2026-09-20: "if the competition rule allow, use external data").

RULE CHECK (docs.crunchdao.com/competitions/competitions/structural-break-real-time, FAQ):
  using the previous challenge's dataset: "Yes, but it is not officially
  encouraged. The dataset was designed for another type of problem, which does
  not match this year's objective of finding not IF, but WHEN."
  pre-trained models: "If you submit a pre-trained models without the training
  code, your model will be ineligible for rewards." (training code may be dead
  code) -> our bundles need the training code added before 2026-10-01.
  runtime 15 h / week; deterministic within 1e-8 on a 10% re-run.

WHY: our limit is the number of SERIES (10,000), not rows (stage 1: all 2.5M
rows -0.0012). The 2025 set has ~10,000 more series from the same organiser
with a known boundary and a break / no-break label; converted to the real-time
format they are real (not spliced) breaks with exact tau.

UPDATE 2026-09-20 05:33: the user downloaded the data into data/ (three
folders). inspect_external.py: the 2025 challenge and the Open Benchmark are
THE SAME 10,001 series (identical values); 29.1% breaks; before-boundary
1,000-2,499 points, after 250-999; value sd 0.095 (this year's 7.15 - our
features are history-standardized); 0 series share a start with this year's.
build_external.py (replaces convert_2025.py; cuts in memory, no big parquet):
9,911 series converted (90 dropped for a too-short history), 481,206 rows
(50 sampled steps per series), positive rows 15.2%; columns = the protocol's
500-column union (439 detector + 61 whitening) + the 27 whitened split tests.
run_model145.py arms: X40 (external replaces the splices), SX40 / SX60
(splices + external pooled). External rows never enter a held fold or the
holdout. Build + pilot started 06:15 (build.log, pilot.log).

RESULT 2026-09-20 (pilot folds 0 / 1, gain vs the current student):
  SX40  splices + external pooled at 40%     +0.0017 / -0.0094
  X40   external replaces the splices (40%)  +0.0052 / -0.0160
-> FAILS: fold 1 collapses (it is the fold that also loses 0.023 without
splices). score_current_on_external.py explains it: the current student scores
only 0.557-0.566 on the 2025 rows (0.62 on this year's held folds; by step
0.53-0.54 for t<64, 0.58-0.59 later), i.e. the 2025 generator is related but
different, and its rows pull the model away from this year's distribution.
Closed as a plain training-row source. Still open (cheap, not run): using the
2025 series only as splice DONORS (their real post-break tails attached to
this year's histories), or as a small share (<= 10%).

EARLIER STATUS 2026-09-20: probe_external.py -> both API keys get "Access Denied" on
the data releases of `structural-break` (id 13) and
`structural-break-open-benchmark` (id 17): the account is not enrolled. The
user was emailed (email_research.txt) to join on the website or to drop
X_train.parquet / y_train.parquet into data/external_2025/.
convert_2025.py is ready (untested until the files exist): cuts each 2025
series so that online length and break position follow this year's empirical
distributions, the same cut for break and no-break series (the cut carries no
label information), history >= 1,000 points.

PLAN once the data is there: (1) check that the 2025 values are standardized
like this year's and that a model trained on 2026 only scores similarly on
the converted series (same generator?); (2) add them to the fitting folds of
the student (never to the held folds / holdout), arms: +25% / +50% / +100%
series; (3) rule, holdout, email before upload.

Other research notes (no code): 2025 winner = stack of XGB/RF on test
features; 2nd place = 6 transforms (z-score, cumsum, dense rank, abs, moving
mean/sd) x statistics + F/Levene/KS + SHAP selection + LightGBM; an
independent 25-method study found heavy regularisation + few features
generalise best and LSTMs/transformers near random. Forum: no method hints.
