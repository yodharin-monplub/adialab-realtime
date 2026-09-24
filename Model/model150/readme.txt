Model150 - sign-flipped re-cuts as a further source of diverse students (2026-09-22).

What worked so far on the sealed holdout was the student side: split-test
features + "move the start" re-cuts + bagging students trained on DIFFERENT
re-cut sets (Submission33, +0.0023). A student's score depends on the re-cut
draw (+/-0.003 per fold), so more diverse sets should help the bag further.
Idea: mirror each re-cut series around its history mean (x -> 2*mean - x). It is
still a real series from the same generator with the same break time, but
every directional feature (up/down CUSUM, signed t, tail_up/tail_down, ...)
sees the opposite direction - a real series "upside down".
Code lives in Model147: build_recuts.py --flip --seed 150 --tag _f150
(+ --split for the 27 split-test columns), run_model147.py arm SRE40f
(splices + this set at 40%, the E-white column sets). No new serving code.
Evaluation: SRE40f alone vs SRE40/b/c, then bags with it (make_bag.py);
holdout only if the bag beats BAG3 (+0.0083 dev) clearly.
RESULTS 2026-09-22 (gain vs the current student, per fold)
  SRE40f (flipped set, seed 150)  +0.0126 +0.0048 +0.0064 +0.0067 +0.0134
          mean +0.0088, s.e. 0.0017, 5/5 - the best single student so far
          (unflipped sets: +0.0071, +0.0078, +0.0045)
  BAG3  (SRE40, b, c)             mean +0.0083
  BAG4F (SRE40, b, c, f)          mean +0.0093  (+0.0134 +0.0046 +0.0066 +0.0088 +0.0130)
One flipped set may be a lucky draw, so two more (seeds 151, 152 -> SRE40g,
SRE40h) are being built and trained (build_f151.log, build_f152.log,
flip2.log); then bags of the flipped students and of all six. A bag goes to
the holdout only if it beats BAG3 by >= +0.002 on the dev folds.

FINAL (2026-09-21, folds 2-4 of g/h finished):
  SRE40g  +0.0096 -0.0006 +0.0029 +0.0035 +0.0089  mean +0.0048, 4/5
  SRE40h  +0.0086 +0.0036 +0.0034 +0.0066 +0.0064  mean +0.0057, 5/5
  BAG3F (f,g,h)            mean +0.0083 (= BAG3)
  BAG6  (SRE40,b,c,f,g,h)  mean +0.0090 (+0.0007 over BAG3, below the +0.002 bar)
Conclusion: flipping is NOT a new lever; SRE40f was a lucky draw. Typical
student ~ +0.006 flipped or not. No holdout, Submission33 stays. Closed.
