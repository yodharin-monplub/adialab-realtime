Model119: Model111 distillation with alpha = 0.75 (target = 0.75*y + 0.25*teacher).

Completes the alpha grid: Model113 (0.25) was slightly below Model111 (0.5) on
raw components (0.622183 vs 0.623373 mean). Teachers are Model111's cached
cross-fitted probabilities for the identical fit rows; everything else is
Model111 (frozen Model84 expressive recipe, 40% aligned synthetic mass).

Run: .venv\Scripts\python.exe -u models\model119\run_model119.py 0 1
Pilot (raw held-fold TS-AUC)
fold      Model119   Model111 (alpha 0.5)
0         0.645640   0.644007   (+0.0016)
1         0.602061   0.607235   (-0.0052)
Mean -0.0018: harder targets help the easy fold and hurt the hard fold, so
alpha grid 0.25/0.5/0.75 -> 0.5 remains best. Confirmation not run. Closed.
