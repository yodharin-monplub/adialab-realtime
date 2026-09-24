Model76: within-time cross-sectional rank aggregation.

The official TSAUC scorer compares predictions only among series at the same
online time step. Model67 averages three probability-like outputs directly,
even though their cross-sectional scales can differ by learner and time. This
study first converts every parent prediction to its percentile rank within
each online step, which preserves every parent's individual TSAUC exactly.
It then combines the frozen Model62/63/64 parents with Model67's fixed
50/25/25 weights using either Borda percentiles or Gaussian normal scores.
The Gaussian aggregate is converted back to a within-step percentile before
blending, so all candidates share a common scale.

Each aggregate replaces 0.25, 0.50, 0.75 or 1.00 of the within-step-ranked
Model67 prediction. These eight choices are frozen before evaluation. Pilot
folds 0-1 select one and require at least +0.0001 mean TSAUC. Confirmation
folds 2-4 then require full and confirmation TSAUC >=0.640, confirmation
gain >=0.0005, 2/3 wins, a positive paired-series bootstrap lower 95% bound,
and positive gains in both fixed series-ID halves.

No learner is fitted and no held or test labels enter the transformation.
All parent choices remain development-informed, so results are exploratory
held-series OOF. The sealed outer holdout and competition labels stay closed.
CPU only; no packaging, upload, or cloud run is automatic.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model76-mpl timeout 2h .venv/bin/python -u models/model76/run_model76.py

