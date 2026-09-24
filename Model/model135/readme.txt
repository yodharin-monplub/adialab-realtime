Model135: parametric break injection as synthetic training data.

Model134 showed bootstrap replicas of real series carry no new break
information, while the splices help because they create breaks that do not
exist in the data. Model135 creates new breaks in a controlled way: each
break-free development series (4,144 donors) gets two replicas with a break
injected at a random tau into its real online values - mean shift, variance
scale, AR(1) dependence change, or distribution-shape change (sign(x)|x|^p) -
with magnitudes drawn from lognormals fitted to the measured real-break
amplitudes (median |mean shift| 0.064 sd / p90 0.21; |log sd ratio| 0.11 /
0.40; |ac1 change| 0.075 / 0.21). Exact tau, realistic difficulty, fold-
tagged by donor. 80 online rows per replica get the 500 causal features.

build_injection.py -> data/inject.npz (~8,300 series, ~660k rows, ~55 min).
run_model135.py (Model134's trainer on this file): arms "replace" (injected
rows instead of splices, 40% mass) and "add" (splices + injected 25%).
Pilot folds 0-1 vs Model111 (0.644007 / 0.607235).
Dataset: 8,388 injected series (2 per break-free donor), 648,318 rows x 500
columns, built in 54 min.
Pilot (raw held-fold TS-AUC of the distilled student)
arm "replace" (injected rows instead of splices, 40% mass)
  fold 0  0.632857 vs 0.644007 (-0.0112)
  fold 1  0.589785 vs 0.607235 (-0.0174)
arm "add" (splices 40% + injected 25% of the remaining real mass)
  fold 0  0.641155 vs 0.644007 (-0.0029)
  fold 1  0.605889 vs 0.607235 (-0.0013)
Conclusion: parametrically injected breaks (mean / scale / AR / shape at
real-break magnitudes) are WORSE training material than splices, both as a
replacement (-0.011/-0.017) and as an addition (-0.003/-0.001). The real
generator produces post-break behaviour that neither simple parametric
shifts nor bootstrap replicas reproduce; only real post-break tails (splices)
teach it. Closed 02:50. Together with Model134: the synthetic-data lever is
exhausted unless the true DGP is known.
