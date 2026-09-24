Model130: synthetic-mass sweep under distilled targets.

The 40% synthetic cell mass (Model62) was tuned with hard labels. Under
distilled targets real rows are less noisy, and both the teacher (0.907 AUC
on splices) and the GRU (+0.29 vs -0.20 mean logit) show the synthetic splices
are far more detectable than real breaks, so the optimal mass may differ.
Arms: synthetic_mass 0.2 and 0.6 (Model111 = 0.4); pilot folds 0-1; everything
else Model111 (frozen Model84 recipe, cached teacher targets).

Run: .venv\Scripts\python.exe -u models\model130\run_model130.py 0 1
Baseline (Model111, mass 0.4): 0.644007 / 0.607235.
Pilot (raw held-fold TS-AUC)
mass    fold 0     fold 1     (Model111 mass 0.4: 0.644007 / 0.607235)
0.2     0.650287   0.598163   (+0.0063 / -0.0091) - fold 1 reverses fold 0 again
0.6     0.635369   0.610777   (-0.0086 / +0.0035) - the mirror image of 0.2
Conclusion: 0.4 remains the right synthetic mass under distilled targets;
neither direction helps on both pilot folds. Closed (21:20).
