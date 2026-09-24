Model166 - is BAGGING still worth it after pooling three re-cut sets?
(2026-09-23)

Two pooled students (5 voters each, Submission39 recipe, trained on the
development series only) with different re-cut set combinations:
  A: "", _s148, _s149      B: _s153, _s154, _s148
SEALED HOLDOUT (student logit bag = 10 voters):
                       student raw   final ws=0.5   final ws=1.5
  A (5 voters)           0.640603      0.637265       0.640380
  B (5 voters)           0.639631      0.637053       0.639974
  BAG A+B (10 voters)    0.640298      0.637171       0.640212
-> bagging no longer helps: one pooled student is as good and costs a third
   of the cloud time. (Bagging mattered when each student saw ONE re-cut set;
   pooling the sets inside one model already removes that variance.)

This run also gives the best estimate of the Submission39 recipe's value:
A at weight 1.5 = 0.640380 on the holdout vs Submission33's 0.634478,
i.e. +0.0059, before the extra gain from also training on the holdout series.
Blend constants: the bundle uses constants_m147_bag3.json whose sd_dist is 1%
smaller, so bundle weight 1.5 = 1.486 in this harness's scale (negligible).
