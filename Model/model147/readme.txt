Model147 - "move the start": re-cut this year's own series into new training
series (from the user's question 2026-09-20 "is there other way to create the
series?", option A).

Idea: the first s online points of a series are clean as long as s is before
the break, so they can be handed to the history:
    history' = history + online[:s]    online' = online[s:]    tau' = tau - s
The result is a fully REAL series from the same generator (nothing glued, no
other dataset) whose break sits at a different online step, so every prefix /
window / step feature changes. One re-cut per development series:
  break series   s uniform in [8, tau - 8]   (tau >= 24)
  no-break       s uniform in [8, n - 16]    (n >= 32)
50 sampled steps per re-cut, 500 union columns (build_recuts.py, the same
offline feature path as the splices). The sealed holdout series are never
re-cut; a re-cut is only used when its PARENT is in the fitting folds of the
held fold, so no held series leaks in through its re-cut.
run_model147.py arms (same student, columns, soft targets, weights):
  R40 re-cuts replace the splices | SR40 / SR60 splices + re-cuts pooled.
Rule: gain > 2 s.e. and >= 4/5 folds, then the sealed holdout.
RESULTS 2026-09-20 (gain vs the current student; 7,788 re-cuts, 375,263 rows,
break share 0.42; per held fold ~225k re-cut rows from fitting-fold parents)
  R40   re-cuts replace the splices   +0.0008 -0.0139           -> fails (the
        splices stay essential, as in stage 1)
  SR40  splices + re-cuts, 40%        +0.0017 +0.0008 +0.0022 +0.0076 +0.0049
        mean +0.0034, s.e. 0.0012, wins 5/5  -> PASSES the rule
SR40 needs no new serving code (same 317 columns, only new fold voters).
Combined arm SRE40 = SR40 data + the Model144 whitened split-test columns
(each fold's pruned column set from models/model144/runs_white/E_fold{k}.npz;
re-cut rows got their 27 split columns with `build_recuts.py --split`, same
seed, row order asserted equal).
  SRE40 +0.0065 +0.0045 +0.0063 +0.0080 +0.0103
        mean +0.0071, s.e. 0.0010, wins 5/5  -> the two gains ADD (features
        +0.0037, data +0.0034); raw student 0.6234 -> 0.6305, the largest
        student gain since distillation.
Dev blend with the chain (select_blend -> models/model112/constants_m147.json):
  +0.0036 at w=.5 (current student +0.0015, split-test student +0.0028).
holdout_eval147.py (ARM147=SRE40; Submission31 with its distilled voters
replaced, weight .5 unchanged, resumable) started 2026-09-20 (~2.2 h),
holdout_SRE40.log. Bar for an upload request: >= +0.001 over Submission31.

SEALED HOLDOUT RESULT (1,667 series, 840,797 rows, 7,963 s):
  Sub28 chain 0.62969 | Sub30 0.63064 | Sub31 0.63222 | SRE40 candidate 0.63376
  gain vs Submission31 +0.00154  -> clears the bar; best holdout score so far.
bootstrap_holdout.py (200 series resamples): mean +0.0016, sd 0.0010,
95% CI [-0.0006, +0.0035], P(gain > 0) = 0.94 (Sub31 had 0.82 when uploaded);
halves +0.0026 / +0.0006; by step 0-64 +0.0009, 64-256 +0.0009,
256-512 +0.0019, >= 512 +0.0023 - positive everywhere.
Deployment: deployment/split_tests_numpy.py = pure-numpy serving of the 27
columns (the cloud bundle has no numba); parity_numpy.py: difference 0.0 on
all 27 columns over 8 series / 4,847 steps, ~1-2 ms per step.
MORE RE-CUT SETS AND BAGGING (2026-09-20 evening; sets _s148 / _s149 = same
builder with seeds 148 / 149, --tag):
  SRE40x3 all three sets in one model   +0.0163 +0.0028 +0.0016 +0.0057 +0.0104
          mean +0.0073, s.e. 0.0027 -> same mean as one set, much noisier: the
          data side saturates at one set per model. Leak audit: no re-cut of a
          held or holdout series is ever used (parents' folds checked row by row).
  SRE40b  one set (_s148)   +0.0133 +0.0024 +0.0023 +0.0090 +0.0121  mean +0.0078
  SRE40c  one set (_s149)   +0.0114 -0.0005 +0.0022 +0.0046 +0.0046  mean +0.0045
  -> a single student depends on the re-cut draw (+/-0.003 per fold); a typical
     one is about +0.0065, the first set was slightly lucky.
  BAG3 = logit mean of SRE40, SRE40b, SRE40c (make_bag.py)
          +0.0126 +0.0035 +0.0054 +0.0087 +0.0113  mean +0.0083, s.e. 0.0017, 5/5
  BAG4 (+ SRE40x3)  mean +0.0089 (not distinguishable from BAG3; 20 voters).
  Dev blend with the chain: BAG3 +0.0037 at w=.5 (SRE40 +0.0036).
BAG3 holdout (15 voters; ARM147=SRE40,SRE40b,SRE40c TAG147=BAG3
CONST147=constants_m147_bag3.json) started late 2026-09-20, holdout_BAG3.log.

BAG3 SEALED HOLDOUT (1,667 series, 8,912 s with 6 workers; 15 voters cost
almost nothing next to the chain):
  Sub31 0.63222 | SRE40 0.63376 (+0.0015) | BAG3 0.63448 (+0.0023)
  bootstrap (200 series resamples): mean +0.0023, sd 0.0010,
  95% CI [+0.0002, +0.0042], P(gain > 0) = 0.99; halves +0.0033 / +0.0012;
  by step 0-64 +0.0018, 64-256 +0.0019, 256-512 +0.0025, >= 512 +0.0028.
-> first holdout gain of the project whose interval excludes zero.
   BAG3 is the Submission32 candidate (build32.py with ARM147=SRE40,SRE40b,SRE40c
   TAG147=BAG3 CONST147=constants_m147_bag3.json). Upload only on the user's word.

BUNDLE = SUBMISSION33 (submissions/submission33 + .zip, 163 MB; "submission32"
was already taken by an unsent GRU-variant bundle of 2026-09-17):
  deployment/build32.py (name kept) with ARM147=SRE40,SRE40b,SRE40c TAG147=BAG3
  CONST147=constants_m147_bag3.json -> 15 student voters, split_tests_numpy.py,
  solution147.py, main.py, training_code/ (30 files + README, prize-eligibility
  rule). validate33.py on 12 holdout series / 5,756 rows through main.infer():
  max |diff| vs the holdout run 3.0e-8 (float32 rounding), replay 0.0,
  runtime 1.147 x Submission31 -> ~5.4 cloud hours. seal33.py: manifest + zip,
  credential scan.
UPLOADED 2026-09-22 on the user's approval ("I allow you to submit to
krittanut"): deployment/push33.py -> submission #6 (id 82565) in project
sympathetic-trout (account bitter-krittanut), platform valid, libraries
verified; deployment/start_cloud33.py -> cloud run 119142 (gcp-cpu, no training,
determinism check). Expected public score ~0.633 (Submission31: 0.6309).
Check with: python models/run_status.py 119142 ; python models/run_score.py 119142

Earlier plan: more re-cut sets (seeds 148, 149 building) to see whether the data side
of the gain scales; bundle (Submission32) = Submission31 with the distilled
voters replaced + SplitTests + the TRAINING CODE the rules require.
