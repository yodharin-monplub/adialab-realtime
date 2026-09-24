Model163 - MORE re-cuts per model once they have soft targets (2026-09-23).

Model147 found that pooling 3 re-cut sets in one model did not help with
HARD labels (SRE40x3 = one set). Since Model158 the re-cuts get oracle-
teacher soft targets, which are far less noisy, so more of them may help.
holdout_recuts3.py (in memory, copy of Model162's harness): the full-dev
recipe (oracle teacher, soft targets on real rows, splices and re-cuts) with
the re-cut sets "", _s148, _s149 pooled in every voter (40% non-real mass).
SEALED HOLDOUT vs the same recipe with one set (FB1 + soft synthetic):
  student raw 0.635119 -> 0.640105  (+0.0050)
  final blend 0.635908 -> 0.637105  (+0.0012)
  = about +0.0026 vs Submission33 with a single 5-voter model
    (Submission36's 3-set bag: +0.0018).
-> with soft targets, more re-cut data keeps paying. Next: 5 sets
(new seeds 153/154, build_sets.log), then the all-series bundle.

5 SETS (seeds 153, 154 built): raw 0.639800 final 0.637045 -> no better than
3 sets (0.640105 / 0.637105). Three pooled sets is the recipe.
SUBMISSION38 = all-series (dev + holdout) version of it: models/model154/
run_all.py arm AP1 (SETS["AP1"] = ["", "_s148", "_s149"], pooled dev AND
holdout re-cuts, soft targets everywhere), 5 voters instead of 15
(bundle 130 MB, ~4.4 cloud hours). Built/validated/sealed with
models/model154/deployment/build35.py ARMS=AP1 DEST=submission38
(parity 4.6e-8, replay 0.0). Emailed the user for approval.
