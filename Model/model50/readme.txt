Model50: recent-onset specialist with causal decayed-peak memory.

The established backbone predicts the cumulative label (a break has already
occurred). Model50 instead asks whether a break first occurred in the most
recent 128 online steps. This gives a specialist for the transition itself.
Its prediction is transformed into a decayed running peak with 256-step
half-life, which carries evidence forward without looking ahead. The peak
margin is added to frozen Model39; it does not replace the strong backbone.

For each held fold, training uses all fitting-fold real rows within 128 steps
after their true break (about 250,000) and an equal number of randomly
selected other real rows. The same 400,000 Model20 aligned-positive synthetic
feature rows are relabelled by their known splice time for the recent-onset
task. Every fitting row and synthetic donor excludes the held fold and the
adjacent stopping fold. The 256 fold-specific causal Model20 features and
depth-3 XGBoost parameters are retained; the classifier uses 800 rounds and
15% mass-preserving synthetic weight share by (online time, recent target).

Only pilot folds 0-1 select an additive weight from 0.05, 0.10, 0.20, 0.40.
If pilot mean gain is below +0.0001, stop. Otherwise fit folds 2-4 and apply
the chosen weight without change. Success requires confirmation and five-fold
means >=0.640, confirmation gain >=0.0005 over Model39, >=2/3 wins, positive
95% series-bootstrap lower bound, and positive gains in two fixed ID halves.

This is exploratory held-series OOF because inherited features and Model39
weights are development-informed. Outer holdout and test labels remain closed.
No cloud training, packaging, upload or submission is automatic. CPU only.

Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model50/run_model50.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model50/run_model50.py report

Research basis: Adams & MacKay, Bayesian Online Changepoint Detection (2007),
https://arxiv.org/abs/0710.3742. Model50 is a heuristic supervised onset
specialist, not an implementation of their exact Bayesian recursion.
