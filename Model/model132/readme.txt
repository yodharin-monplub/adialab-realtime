Model132: cross-family teacher average (tree-LLR teacher + bi-GRU teacher).

Model117 averaged same-family (LightGBM) teachers over seeds: no gain. The
Model109 tree teacher (privileged onset-likelihood features) and the Model131
bidirectional GRU teacher are different model families with different biases;
their average may be a better soft target than either. Real-row targets:
0.5*y + 0.5*mean(tree prob, bi-GRU prob); both fold-clean for held fold k on
the identical frozen fit rows. Everything else is Model111. Pilot folds 0-1.
This is the last zero-cost variant in the distillation family to be tested.

Run: .venv\Scripts\python.exe -u models\model132\run_model132.py 0 1
Baselines: Model111 0.644007 / 0.607235; Model131 0.640460 / 0.610069.
Pilot (raw held-fold TS-AUC): fold 0 0.640996 vs Model111 0.644007
(-0.0030); fold 1 0.611302 vs 0.607235 (+0.0041). Mean +0.0005.
The opposite-sign fold split of Models 119/130/131 again: fold 1 prefers
"different" targets, fold 0 the LLR-tree teacher. Not deployable as such;
see the bagging check in models/SESSION_RESULTS_2026-09-17.md. Closed 22:32.
