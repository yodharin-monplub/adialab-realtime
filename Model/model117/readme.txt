Model117: Model111 distillation with seed-averaged teachers.

Hypothesis: teacher *noise* (not teacher strength, which Model110 tested)
limits the distillation gain. For each held fold k and fitting fold j, three
teachers are trained with different seeds, row samples and column fractions
(0.5/0.4/0.6) and their probabilities averaged before forming the soft target
0.5*y + 0.5*teacher. Fold-exclusion rules, fit rows, 40% aligned synthetic
mass, 317 fold columns and the expressive LightGBM student are Model111's.
Baseline: Model111 raw components 0.644007/0.607235/0.627220/0.608750/0.629653.

Run: .venv\Scripts\python.exe -u models\model117\run_model117.py 0 1
Pilot (raw held-fold TS-AUC)
fold   teachers x3 (fit rows)              Model117   Model111
0      0.71219 / 0.69302 / 0.70700         0.643836   0.644007   (-0.0002)
       (single-seed: 0.70390/0.69005/0.69824)
1      0.73034 / 0.69229 / 0.71396         0.607543   0.607235   (+0.0003)
       (single-seed: 0.73019/0.68711/0.71787)
Seed averaging reduces teacher noise (up to +0.008 teacher TS-AUC) but the
student does not improve (mean +0.0001 on the pilot). Teacher noise is not
what limits distillation. Confirmation folds not run. Closed.
