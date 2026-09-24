Model51: history-whitened, time-integrated Bayesian evidence.

Model50's onset target harmed the pilot. A metric audit showed that online
steps 160-640 carry about 69% of the official positive-negative pair weight.
Model51 therefore uses long-window evidence rather than early-onset spikes.

For each series, an AR(5) model is fitted only on its clean history using the
existing d3 history estimator. Historical residual ranks define an empirical
CDF; inverse-normal scores whiten both historical and online residuals.
Four causal innovation channels (location, scale, absolute tail, lag-one
dependence) are history-centred and history-scaled. At every online step,
Gaussian-prior Bayes factors for an unknown persistent shift are computed over
six suffix lengths (16-512) and two prior shift variances. Log-mean-exp
integrates across possible break times, channel and amplitude. The resulting
score is robustly centred/scaled against the same aggregate statistic on
break-free history. This is a heuristic online Bayes-factor ensemble, not a
full Bayesian changepoint posterior.

No labels fit the evidence score. It is added to the frozen Model39 margin at
pilot-selected weights 0.05, 0.10, 0.20, 0.40. Pilot folds 0-1 must improve
their mean by >=0.0001 before folds 2-4 are evaluated. Success requires
confirmation and full five-fold means >=0.640, confirmation gain >=0.0005,
>=2/3 wins, positive 95% series-bootstrap lower bound, and positive gains in
both fixed series-ID halves. The inherited Model39 model is exploratory and
development-informed; outer holdout and test labels remain closed.

CPU only, four feature processes. No cloud training, packaging, upload or
submission is automatic. Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model51/run_model51.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model51/run_model51.py report

Motivation: Adams & MacKay, Bayesian Online Changepoint Detection,
https://arxiv.org/abs/0710.3742, and the CrunchDAO community's history-
whitening plus break-time integration findings,
https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206.
