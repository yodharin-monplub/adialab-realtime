Model151 - distilled re-cuts (2026-09-21).

What worked on the sealed holdout: (1) distillation - real rows are trained on
soft targets 0.5*y + 0.5*teacher, where the teacher (Model109/111) sees the
whole online segment; (2) Model147 re-cuts - real series with the start moved.
But in Model147 the re-cut rows carry HARD labels only.
A re-cut is the SAME values as its parent with the history/online boundary
moved, so the teacher's privileged features can be computed on it exactly
(teacher_recuts.py; with shift 0 it reproduces the Model109 cache to <1e-7).
For held fold k, a re-cut whose parent is in fold j gets the prediction of
Model111's saved fold{k}_teacher{j}.txt (never trained on folds k, k+1, j) -
the same cross-fitting as the real rows. Everything else = SRE40.

Files
  teacher_recuts.py  -> models/model147/data/recuts{tag}_teacher.npy
                        (python models/model151/teacher_recuts.py base _s148 _s149)
  run_model151.py    arms SRD40 / SRD40b / SRD40c (re-cut sets "", _s148, _s149)
                     python models/model151/run_model151.py SRD40 0   (one fold per call)
  srd.log            results; runs/ = fold models + OOF predictions
No new serving code: the deployed student has the same inputs (only new
fold-voter weight files).

RESULTS (gain vs the hard-label twin SRE40 on the same data and seed)
  fold 0: 0.653499 vs 0.650476 = +0.0030  (vs current student +0.0095)
  fold 1: -0.0027   fold 2: +0.0014   fold 3: -0.0027   fold 4: -0.0007
  SRD40 vs hard-label twin: mean -0.0003, 2/5  -> FAILS.
  (re-cut teacher TS-AUC ~0.60 vs its 0.686-0.704 on real rows (fold{k}_teacher.json): the teacher is weak
  on re-cuts, so its targets add noise rather than remove it.)
  As a 4th bag member (BAG3D = BAG3 + SRD40): +0.0085 vs BAG3 +0.0083 -> noise.
  SRDS (soft splice targets, coded in run_model151.py) not run: same mechanism.
CLOSED 2026-09-21. Submission33 stays.
