Model122: seed-bag of the distilled student (Model111 recipe, extra seeds).

Student-side variance reduction, complementary to Model117 (teacher side).
Two additional LightGBM seeds (random_state offsets 1001, 2002) of the exact
Model111 student are trained on the identical fit rows, targets, columns and
weights; the three-seed logit average is compared with Model111 alone.
The recipe seed must stay fixed because it also selects the frozen fit rows,
so only LightGBM's random_state is offset through a shim on Model111's `lgb`
(the first attempt changed the recipe seed and failed the fit-row check).

Run: .venv\Scripts\python.exe -u models\model122\run_model122.py 0 1

Pilot (raw held-fold TS-AUC)
fold   Model111   seed+1001   seed+2002   bag3
0      0.644007   0.643028    0.645511    0.645805   (+0.0018)
1      0.607235   0.603787    0.605732    0.606770   (-0.0005)
Seed-to-seed spread of the student is about +/-0.001 (fold 0) and +/-0.0035
(fold 1). Against the *average* seed the bag gains about +0.0014 on both
folds, but Model111's fold-1 seed was a lucky draw, so vs the deployed voter
the pilot mean is only +0.0007 (and roughly half of that survives the top-
level blend).

Confirmation folds 2-4 (raw held-fold TS-AUC)
fold   Model111   seed+1001   seed+2002   bag3
2      0.627220   0.622890    0.620018    0.624725   (-0.0025)
3      0.608750   0.607295    0.605439    0.608159   (-0.0006)
4      0.629653   0.623434    0.624382    0.627414   (-0.0022)
Five-fold mean of bag3 minus the deployed Model111 seed: -0.0008. The
deployed seed is above both extra seeds on 4/5 folds (a favourable draw that
is already baked into Submission30's cloud score). A seed-bagged Submission31
would not improve on Submission30; not built. Closed.
