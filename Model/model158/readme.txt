Model158 - soft teacher targets for the SYNTHETIC rows (2026-09-22).

Until now only the real rows were trained on 0.5*y + 0.5*teacher; splices and
re-cuts kept hard 0/1 labels. Model151 tried soft re-cut targets with the old
teacher and failed (that teacher scored 0.60 on re-cuts). The Model157 oracle
teacher is a model-based posterior of the break position, so it may transfer.

Teacher for held dev fold k: Model152's 4-fold oracle teacher
full_teacherB{k}.txt (never saw fold k); inputs = fold-0 student columns +
25 Model109 + 7 oracle features of the glued / re-cut series
(run_model158.py feats; splice union columns from Model155).
  teacher TS-AUC on the synthetic rows: splices 0.92, re-cuts 0.75
  (old teacher on re-cuts: 0.60)

Student SRBS40 = SRB40 (Model157) + soft synthetic targets, vs SRB40:
  fold 0 +0.0005  1 +0.0025  2 +0.0044  3 +0.0012  4 +0.0031
  mean +0.0023, 5/5 (~3.5 s.e.)
SEALED HOLDOUT (run in memory - the disk was full; same one-change test):
  student raw 0.62950 -> 0.63152 (+0.0020)
  final blend 0.63449 -> 0.63489 (+0.0004)
On top of the full-data oracle refit (FB1, set "", in memory):
  student raw 0.63272 -> 0.63512 (+0.0024)
  final blend 0.63556 -> 0.63591 (+0.0004)
-> TRANSFERS and stacks with the Submission36 recipe.
Folds 1-4 models were not saved (disk full); fold 0 is in runs/.
Used in the final recipe as Model154 run_all.py arms AS1/AS2/AS3 (Model159).
