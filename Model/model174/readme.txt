Model174 - does a MIRRORED re-cut set earn a place among the pooled sets?
(2026-09-25)

WHY
Model150 measured the mirrored re-cut set as the best single tree student in this
project: +0.0088 dev, 5/5, against +0.0071 / +0.0078 / +0.0045 for unflipped
draws. That was the Submission33-era recipe. The recipe then changed (oracle-
teacher soft targets on synthetic rows, Model158; three re-cut sets pooled,
Model163) and the mirrored sets were never carried across. Model163 only pooled
UNFLIPPED redraws and found five no better than three - which says nothing about
a set that reverses DIRECTION rather than redrawing shifts.

FILES
teacher_flip.py   models/model151/teacher_recuts.py refuses flipped sets ("not
    supported (not needed)"): it rebuilds each re-cut from the pool WITHOUT the
    mirror, so its features would describe the unflipped series while the rows
    describe the flipped one. This builds the two missing inputs with the mirror
    applied exactly as models/model147/build_recuts.py --flip does it
    (centre = 2*mean of the ORIGINAL history, before the re-cut extension):
      models/model147/data/recuts_f150_teacher.npy   (375,991 x 25)
      models/model158/data/recut_oracle_f150.npy     (375,991 x 7)
    Both built in under a minute.
run_model174.py   Model164's student (the dev-fold twin of the deployed recipe)
    with ONLY the pooled set list, the arm name and the output directory changed,
    so its numbers compare directly with models/model164/runs/SRP40_fold*.npz.
      python models/model174/run_model174.py FLIP3 0 1 2 3 4
    ARMS: FLIP3 ["", "_s148", "_f150"] · FLIP4 [+ "_s149"] · FLIP3b ["_f151"]

SANITY CHECK FIRST
The oracle teacher transfers to the mirrored re-cuts: TS-AUC 0.7385 on re-cut
rows, 0.9199 on splices. (Model151 failed precisely because the OLD teacher
scored only 0.60 on re-cuts, so this check is not a formality.)

RESULT - ARM FLIP3 (one unflipped draw swapped for the mirror), 1,126,754 pooled
re-cut rows, dev folds, vs SRP40 = the deployed three-set pool:
  fold 0   0.661538  vs 0.660905   +0.000633
  fold 1   0.608575  vs 0.608836   -0.000261
  fold 2   0.640097  vs 0.639749   +0.000348
  fold 3   0.620219  vs 0.620040   +0.000179
  fold 4   0.646963  vs 0.645669   +0.001294
  mean gain +0.000439 +/- 0.000258, wins 4/5
-> FAILS the rule: 4/5 folds but only 1.7 s.e. (the bar is 2 s.e. = +0.000517).
Consistent, and far too small to matter: +0.0004 on the student is roughly
+0.0002 in the blend, an order of magnitude inside the holdout noise (+/-0.002)
and inside the holdout->cloud offset (-0.0014).
READING: the mirrored set's old +0.0088 came from being the ONLY extra set. Once
three sets are pooled with soft targets, the diversity a mirror adds is already
there. FLIP4 was not run: Model163 showed a fourth/fifth set adds nothing, and
the expected effect here is smaller than the noise we could measure it with.
-> The tree student is confirmed saturated, as the neural student was tonight
(Model172/173). Data levers on BOTH halves are now closed.
