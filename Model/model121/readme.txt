Model121: age-softened hard label combined with the learned teacher.

    target = 0.5 * y * g(age) + 0.5 * teacher,   g(age) = 1 - 0.5*exp(-age/32)
Model120 tests g(age) alone; Model121 tests whether the two softenings are
complementary. Teacher probabilities are Model111's cached cross-fitted
values (identical fit rows); everything else is Model111.

Run: .venv\Scripts\python.exe -u models\model121\run_model121.py 0 1
Pilot (raw held-fold TS-AUC)
fold      Model121   Model111
0         0.643400   0.644007   (-0.0006)
1         0.606303   0.607235   (-0.0009)
Result: age-softening is not complementary to the learned teacher (slightly
worse on both folds). Confirmation not run. Closed.
