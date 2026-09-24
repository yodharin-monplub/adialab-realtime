Model56: absorbing-break evidence filter on Model39's causal margin.

Model39's current margin already contains a short-memory causal smoother,
but its trajectory may hold evidence of a persistent break over a longer
horizon. Model56 runs a fixed two-state Bayesian-style recursion on each
series independently: the prior probability of a break increases by a
constant hazard 0.0009 at every online observation; the current Model39
margin supplies a tempered likelihood increment 0.08*(margin+0.25). The
output is the posterior log-odds excess over the hazard-only trajectory,
divided by sqrt(online age). This is a heuristic absorbing-state filter,
not the exact Bayesian online change-point detector of Adams & MacKay.

Only history through the current online point is used. No label, future
observation, final horizon or other test series is read by the transform.
One exact prefix-invariance test is run before scoring. The candidate adds
the excess-evidence score to the unchanged Model39 margin at fixed weights
0.5, 1, 2 or 4. Pilot folds 0-1 select one weight; if mean gain is below
+0.0001, stop before inspecting confirmation folds 2-4. Otherwise require
confirmation and full held-series OOF TSAUC >=0.640, confirmation gain
>=0.0005, at least 2/3 wins, positive 95% paired series-bootstrap lower
bound, and positive gains in both fixed series-ID halves.

This is exploratory because Model39 itself and the choice of filtering
direction are development-informed. The sealed outer holdout and test
labels remain closed. No cloud run, packaging, upload or submission.
CPU only; no model fitting.

Research motivation: Adams & MacKay (2007), Bayesian Online Changepoint
Detection, https://arxiv.org/abs/0710.3742.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model56-mpl .venv/bin/python -u models/model56/run_model56.py run
