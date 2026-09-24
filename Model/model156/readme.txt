Model156 - calibrated Bayesian break evidence (Discord-brief recipe), 2026-09-22.

bayes_evidence.py: history-only AR whitening -> history-ECDF normal scores u;
for every step, Bayes factors of "break in the last m points" vs no break,
with amplitude priors mixed within each channel and a UNIFORM prior over the
break position (grid cells weighted by the number of positions they cover):
  scale  v ~ InvGamma(a, a-1), a in {100,30,12,6,3}
  mean   mu ~ N(0, s^2), s in {.05,.12,.30}
  dep    AR(1) rho in {+/-.06, .12, .20, .35}
3 columns (be_scale, be_dep, be_mean). Leak checks passed (prefix invariance,
future perturbation). numba; ~25 s per 400k rows.
Calibration on our dev breaks (normal-score stream), median / p90:
  |log sd ratio| .097/.317   |mean shift| .067/.195   |ac1 change| .088/.233
  -> matches the brief's grids (.109, .064, .075 medians); grids kept.
Single-column TS-AUC (dev / holdout): scale .585/.568, dep .516/.517,
  mean .515/.517.

Student BE40 = SRE40 + the 3 columns (never pruned), run_model156.py:
  fold  0 +0.0020  1 +0.0009  2 +0.0016  3 +0.0023  4 +0.0019
  mean +0.0017, s.e. 0.0002, 5/5; evidence = 5-8% of split gain.
SEALED HOLDOUT (offline_be.py, same one-change comparison):
  student raw  SRE40 0.62702 -> BE40 0.62772  (+0.0007)
  final blend  SRE40 0.63376 -> BE40 0.63394  (+0.0002)
-> does NOT transfer (dev folds are reused / optimistic; feature gains of
this size have not survived the holdout since Model144). CLOSED.
Data kept: data/real.npy, syn_fold{k}.npy, recuts.npy, holdout.npy.
