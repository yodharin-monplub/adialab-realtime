Model175 - a DIFFERENT MODEL FAMILY as a fourth blend component (2026-09-25).

WHY
Every part of the deployed blend is a gradient-boosted tree or a GRU. What the
blend has always paid for is DIVERSITY: the GRU part scores only ~0.62 alone yet
is worth +0.0016 in the blend. The chain does contain an ExtraTrees and a
RandomForest, but they were trained in August on development series only with
HARD labels - before the two levers that actually mattered were found (oracle-
teacher soft targets, +0.007; all-series training, +0.007). No family other than
LightGBM and the GRU has ever been given the current recipe.

HOW
run_model175.py runs Model164's student assembly UNCHANGED - same rows, same 40%
synthetic mass, same soft targets, same columns - and replaces only the estimator,
by swapping the LGBMRegressor factory inside Model164's namespace for an adapter
(fit / predict / booster_.save_model). So exactly one thing differs: the learner.
  python models/model175/run_model175.py ridge 0 1 2 3 4        (~45 s per fold)
  python models/model175/run_model175.py extratrees 0 1         (~30-40 min per fold)
blend4.py then asks the only question that matters - does it add anything to
Submission42's three-part blend? - searching its weight on pilot folds 0-1 and
reporting confirmation folds 2-4 separately, the project's standard protocol.

RESULT - RIDGE (alpha 10, standardized features), all five folds
  standalone TS-AUC per held fold: 0.625016 fold 0, mean 0.6093 over folds 1-4
  (the GBDT student is 0.6609 on fold 0; the GRU part about 0.6377)
  correlation with the student +0.236, with the nets +0.194  <- genuinely
  decorrelated, which is what a fourth component is supposed to bring
  as a fourth component (student 1.5, nets 1.0 fixed):
    w4=0.1  pilot +0.00004  confirm -0.00000   2/5
    w4=0.3  pilot +0.00010  confirm -0.00003   2/5
    w4=0.5  pilot +0.00012  confirm -0.00010   2/5   <- pilot-selected
    w4=1.0  pilot +0.00006  confirm -0.00040   1/5
  -> REJECTED. The useful lesson: a component can be strong-ish (0.61-0.63) AND
  decorrelated from both existing parts and still add nothing. Being wrong in
  different places is not the same as being right where the others are wrong.

EXTRATREES: pilot folds 0-1 running (same recipe, 150 trees, min_samples_leaf 50,
max_features 0.3). It is the strongest remaining family-level idea because it is
the one the chain uses but never with soft targets or all-series data. Decision
rule as always: the pilot-selected weight must also win on folds 2-4, at >= 4/5
overall, before anything is built.

RESULT - EXTRATREES (150 trees, min_samples_leaf 50, max_features 0.3), pilot
folds 0-1, same rows / targets / columns as the GBDT student
  standalone, held fold      XTREES     GBDT student
    fold 0                   0.646900   0.660905
    fold 1                   0.613719   0.608836   <- BEATS the student here
  fold 0 took 2,303 s, fold 1 2,096 s (14 GB resident)
  correlation with the student +0.466, with the nets +0.372
  as a fourth component (student 1.5, nets 1.0 fixed), pilot folds 0-1:
    w4=0.1 +0.00005 · w4=0.2 +0.00007 · w4=0.3 +0.00007 (pilot-selected)
    w4=0.5 +0.00002 · w4=0.75 -0.00010 · w4=1.0 -0.00029
  -> REJECTED AT PILOT. Folds 2-4 were NOT run: the effect is +0.00007, about 30x
  smaller than the holdout noise (+/-0.002), and the pilot gate exists precisely
  to stop a 1.75 h confirmation run of an effect that cannot matter.
  Interesting all the same: it is a genuinely strong learner on this data and it
  beats the boosted student on the anomalous fold 1, and STILL adds nothing.

WHAT THE TWO FAMILY EXPERIMENTS TOGETHER SAY (and why they close Model172's
"option B" as well)
  ridge      standalone ~0.61-0.63, correlation +0.24/+0.19, adds +0.00012 pilot
             and -0.00010 on confirmation
  extratrees standalone 0.615-0.647, correlation +0.47/+0.37, adds +0.00007
  A fourth component adds nothing whether it is weakly or strongly correlated
  with what is already there. The three deployed parts already span the signal
  this feature space carries; being wrong in different places is not the same as
  being right where the others are wrong.
  This also kills the last untried structural idea - retraining the old chain on
  all series with soft targets. That would make the chain MORE like the student
  (same features, same targets, same data), i.e. it would remove the very
  difference that earns the chain its weight, while costing 8-14 h on hash-pinned
  August code. It is not worth running, on evidence rather than on cost.
