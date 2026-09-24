Model142 - causal post-processing of the final score (user's question 02:06,
"is there any kind of post processing that will help?").

TS-AUC ranks series against each other at the same online step, so a
per-step monotone recalibration cannot change it; only series-specific
temporal operations can. Sweep on the Submission31 OOF score (chain + .5
distilled + .5 GRU, 5 dev folds, postproc.py, no refitting), 74 variants:
  EMA smoothing of the logit  half-life 2-32, weight .3/.6/1
  momentum  x + g (x - EMA)   g .25/.5/1
  decaying max  max(x, m - delta), delta .01-.3, weight .1/.3/.6
  own-baseline subtraction  x - a mean(x[first k steps]), k 8-64, a .25/.5/1
  own-history rank  x + w P(past score < x), w .25/.5/1
Result (score.txt): nothing beats noise. Best = baseline k=16 a=.25 at
+0.0008 +/- 0.0011 (4/5, fold 0 -0.003; a=.5 turns negative). Smoothing
hurts monotonically (h=16 full: -0.006 - the chain already carries a
rolling-max-128 step), momentum hurts (-0.0009), decaying max is neutral at
w=.1 and negative above, own-history rank -0.0003. Full cummax was -0.04
(Model136). Closed: the score trajectory is already as good as its inputs.
