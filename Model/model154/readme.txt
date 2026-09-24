Model154 - ALL-data students for the final bundle (dev + holdout = 10,000
labelled series), 2026-09-21.

The student recipe was chosen and verified on the sealed holdout in Model152
(full-dev refit BAGF3 +0.0013 vs Submission33, 95% [+0.0005, +0.0023]).
This step adds the 1,667 holdout series (+20% series) to that exact recipe.
After this, the holdout can no longer score these models (in-sample), so
every decision is made BEFORE this step. Standard practice for a final fit.

1. prepare_holdout.py
   rows    168,500 sampled holdout rows (Model20 20% hash rule) with the 500
           union columns + 27 split tests from Model152's streaming pass.
           Parity: streaming vs offline dev cache = 0.0 difference (5 dev
           series, all 527 columns), so holdout rows are the same features.
           Plus 25 Model109 teacher features per row.
   recuts  Model147 re-cuts of the holdout series (seeds 147/148/149 -> "",
           _s148, _s149), ~1,557 re-cuts / 75k rows per set, 500 + 27 cols.
           Sanity: medians vs dev re-cuts within 0.09 IQR; positive share
           0.301 vs 0.304.
2. run_all.py
   teacher  one teacher on dev rows only (Model111 settings) -> soft targets
            for the holdout rows, out-of-sample: TS-AUC 0.7146.
   ALL/ALLb/ALLc  15 voters = Model152 FULL recipe + holdout rows + holdout
            re-cuts (fold v's columns, seed, splices).
     python models/model154/prepare_holdout.py rows
     python models/model154/prepare_holdout.py recuts 147 base   (148 _s148, 149 _s149)
     python models/model154/run_all.py teacher
     python models/model154/run_all.py ALL      (ALLb, ALLc)

3. check_holdout_rows.py - out-of-sample test of the holdout pathway
   (FULL recipe on 4 dev folds, with / without all holdout material, same
   code path as run_all.py, scored on the 5th dev fold, check.log):
     fold 0  0.659935 -> 0.660811  +0.0009
     fold 1  0.624409 -> 0.628602  +0.0042
     fold 2  0.649423 -> 0.655872  +0.0064
   3/3 positive, mean +0.0038 per single voter: no alignment bug, and the
   extra 1,667 series help as the learning curve predicted.
4. deployment/ (build35.py, validate35.py, seal35.py, TRAINING_README.txt)
   Submission35 = Submission31 chain + GRUs + the 15 ALL voters, BAG3 blend
   constants, same serving code as Submission33 (models/model147/deployment),
   35 training-code files.

RESULTS
  15 ALL voters trained (all.log). Holdout score of ALL3 is IN-SAMPLE
  (0.858 raw) - meaningless as a score, used only as parity reference.
  Expected cloud gain vs Submission33: at least BAGF3's +0.0013 (holdout-
  verified recipe) plus the extra-series effect.
