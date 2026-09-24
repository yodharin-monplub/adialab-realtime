Model125: soft targets for the synthetic splice rows as well.

Model111 distils only the real rows; the 400k aligned synthetic rows (40% of
the training mass) kept hard labels. Here the fold's three saved teachers
(Model111 runs, trained on folds outside {k, k+1, j}) also score the synthetic
series: privileged features are computed from each spliced (history, online)
pair with models/model109/teacher_features.py, the teachers' probabilities are
averaged, and the synthetic targets become 0.5*y + 0.5*teacher too. Fit rows,
synthetic rows, 40% synthetic cell mass, 317 fold columns, weights and the
expressive LightGBM recipe are Model111's.

Run: .venv\Scripts\python.exe -u models\model125\run_model125.py 0 1

Teacher TS-AUC on synthetic rows: 0.907 (fold 0), 0.894 (fold 1) - splices are
far more detectable than real breaks, so the soft targets differ strongly from
the hard labels near the splice point.

Pilot (raw held-fold TS-AUC)
fold      Model125   Model111
0         0.649375   0.644007   (+0.0054)  largest single-fold gain of the session
1         0.603271   0.607235   (-0.0040)
The pre-set gate (fold 1 >= +0.001) failed; folds 2-4 were run as exploratory:
2         0.627408   0.627220   (+0.0002)
3         0.609888   0.608750   (+0.0011)
4         0.629028   0.629653   (-0.0006)
Five-fold mean +0.0004: noise-level. Soft targets on the synthetic rows do not
add to the real-row distillation (the fold-0 jump was not reproducible).
Closed; nothing deployed.
