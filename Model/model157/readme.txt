Model157 - a stronger teacher: Bayesian ORACLE features (2026-09-22).

Lesson of the project so far: feature additions to the causal student do not
survive the sealed holdout (Model144 split tests alone +0.0005, Model148,
Model156 +0.0002), but DATA and TARGET changes do (distillation, re-cuts,
more series). So improve the targets: the student learns 0.5*y + 0.5*teacher,
and the teacher sees the whole online segment.

oracle_bayes.py: for every candidate break position k over the WHOLE online
segment, Bayes factor of "break at k" vs "no break" on the normal-score
stream (history-only whitening), with Model156's calibrated amplitude priors
(scale / dependence / mean, and their 1/3 mixture), P(break)=1/2 and a
uniform prior on k -> the posterior of the label itself:
  ob_cdf (= P(tau <= t | all data)), per-channel cdfs, P(break), log odds,
  posterior mean age.  ~1.5 min for all 4.2M dev rows (vectorised numpy).
  Single-column TS-AUC (all dev rows): ob_cdf 0.673, ob_cdf_scale 0.674,
  dep 0.609, mean 0.595 (old best privileged column z_cdf ~0.655 on a sample).

run_model157.py: Model111's cross-fitted teachers re-trained with the old 25
privileged features + these 7 (same rows, seeds, settings) -> teacherB:
  better than the old teacher in 14 of 15 fits, e.g. fold 0: .719/.699/.724
  vs .704/.690/.698.
Student SRB40 = SRE40 with only the soft targets changed:
  dev  fold 0 +0.0066  1 -0.0025  2 +0.0030  3 +0.0002  4 +0.0010
       mean +0.0017, s.e. 0.0015, 4/5 (not significant on dev)
  SEALED HOLDOUT (offline_holdout.py, one change vs SRE40):
       student raw 0.62702 -> 0.62950 (+0.0025)
       final blend 0.63376 -> 0.63449 (+0.0007) = BAG3's level with 1 set
  -> TRANSFERS (target changes keep transferring; feature changes do not).

Next: combine with the full-data refit (Model152 run_full.py, ORACLE=1 ->
4-fold teachers with oracle features, arms FULLB/FULLBb/FULLBc), compare
with BAGF3 (0.63578) on the holdout; then the all-series version (Model154)
for the final bundle.
  python models/model157/oracle_bayes.py real
  python models/model157/run_model157.py teachers 0 ; ... student 0

TEACHER CAPACITY (teacher_probe.py; variant C in run_model157.py, TVAR=C)
  one slot (fold 0, j=2): base .7190 | 900k rows .7243 | 1500 trees .7226 |
  127 leaves .7245.  Variant C = 127 leaves + 900k rows, fold 0:
  .7253/.6992/.7222 vs teacherB .7190/.6993/.7239 (mean +0.0015 only).
  Student SRC40 fold 0: 0.65262 vs SRB40 0.65704 (-0.0044) -> a bigger,
  more confident teacher gives WORSE targets. Stopped after fold 0; teacherB
  settings stay.

FULL-DATA REFIT WITH THE ORACLE TEACHER (Model152 run_full.py ORACLE=1)
  4-fold oracle teachers .741/.696/.730/.702/.727 (old 4-fold .732/.699/
  .717/.697/.726). Voters FB1 (set ""; the name FULLB collided with FULLb on
  case-insensitive Windows paths - that run was skipped, nothing was
  overwritten), FULLBb, FULLBc.  Sealed holdout:
    FULLBb vs FULLb   student .63651 vs .63441 (+0.0021), final +0.0007
    BF3 = FB1+FULLBb+FULLBc  final 0.63627 vs Submission33 0.63448
      gain +0.0018, bootstrap 95% [+0.0009, +0.0028], P>0 1.00,
      halves +0.0027/+0.0009, every step range positive
    (BAGF3, the refit without the oracle teacher: +0.0013)
  Blend weights re-checked on dev (weights_eval157.py with SRB40): the
  pilot-selected ws=1.0 fails confirmation -> 0.5/0.5 kept.
  All-series version: Model154 run_all.py arms AB1/AB2/AB3 (hold_teacherB:
  holdout rows out-of-sample TS-AUC 0.7271 vs 0.7146).
