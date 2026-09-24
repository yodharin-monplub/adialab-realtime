Model120: age-softened labels as a hand-built, tau-only teacher.

Hypothesis: distillation helps because post-break rows at small age since
the break are nearly indistinguishable from pre-break rows, so their hard
label 1 is mostly noise. Test: replace the learned teacher by
    target = y * (1 - 0.5*exp(-age/32)),  age = t - tau_index
(0.5 at the break, 0.82 at age 32, 0.93 at age 64). tau is used only to build
training targets. Everything else is Model111's frozen Model84 recipe.

Run: .venv\Scripts\python.exe -u models\model120\run_model120.py 0 1

Pilot (raw held-fold TS-AUC)
fold      Model120   Model111 (learned teacher)   Model84 (hard labels)
0         0.642620   0.644007                     0.643302
1         0.600514   0.607235                     0.599595
Result: age-softening alone performs like hard labels (-0.0007 / +0.0009) and
does NOT reproduce the distillation gain (-0.0014 / -0.0067 vs the learned
teacher). The learned teacher carries information beyond age since the break
(e.g. series-level detectability). Confirmation not run. Closed.
