Model89: channel-separated, tau-integrated Bayesian evidence.

This is the strongest legitimate modeling suggestion in the Discord research
brief. It explicitly avoids public-series memorization. Model51's earlier
Bayes probe combined four standardized shift channels, only used widths
16--512, assigned equal mass to geometric grid nodes, and blended its raw
score directly into Model39. Its pilot failed badly. Model89 changes each of
those material choices rather than repeating that experiment.

Existing Model57 history-only AR/BIC, forward EWMA, and empirical-normal
whitening is reused. Two causal evidence columns remain separate:

1. scale: an exact inverse-gamma variance Bayes factor mixed over shapes
   100, 30, 12, 6 and 3 with b=a-1;
2. dependence: an exact conditional Gaussian AR(1) likelihood ratio mixed
   over rho = +/-0.06, +/-0.12, +/-0.20 and +/-0.35.

Candidate post-break ages are 1,2,3,4,6,...,768 plus the current partial
cell. Each node receives the number of individual tau values it represents,
so the integration prior is uniform over all observed candidate break times
and its mass sums exactly to online age. The two evidence values are appended
as supervised LightGBM inputs instead of used as an uncalibrated direct score.

Scale-only, dependence-only and both-channel arms retain Model84's exact
400,000 real + 400,000 aligned-splice rows, 40% synthetic TSAUC-cell mass,
expressive learner, causal stack, and Model67/84 baseline. Pilot folds 0--1
select an arm and blend weight from 0.25, 0.50, 0.75; a +0.0001 gain is
required before folds 2--4. Promotion requires full and confirmation TSAUC
>=0.640, confirmation gain >=0.0005, 2/3 wins, positive paired-series
bootstrap lower 95%, and positive gains in both fixed ID halves.

CPU only, resumable caches/checkpoints, no cloud activity. Public/test labels
are never used. Prefix invariance, candidate-prior mass, finite output, parent
hash, and baseline reconstruction checks run before fitting.

Research brief: ../DISCORD_NEXT_MODEL_BRIEF.md
Prior comparison audit: audit_amplitudes.py (diagnostic only; it cannot change
the predeclared prior grid).

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model89-mpl .venv/bin/python -u models/model89/run_model89.py
