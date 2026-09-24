Model27 — family-matched contrastive augmentation and history context
====================================================================
Goal: materially improve held-series OOF TS-AUC toward 0.640 without GPU use.
Run/resume: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model27/run_model27.py run

Motivation
----------
Model20's arbitrary post-break splices produced the largest recent cloud gain,
but median-only alignment can make donor seams unrealistically easy. Model27
creates paired examples with the same recipient prefix and matched donor:

- Positive: recipient clean prefix + donor's true post-break tail.
- Hard negative: identical recipient prefix + a clean donor-history tail.
- Both tails receive robust location/scale transport into the recipient frame.
- Donors are matched by 13 clean-history descriptors and must come only from
  fitting folds. Validation, adjacent stopping, and reserved outer series are
  forbidden as either recipient or donor.

The same 13 inference-safe history descriptors are forced into the new model:
history length, four autocorrelations, differenced scale, volatility
persistence, skew, kurtosis, tail rate, sign persistence, trend, and low
frequency energy. They are intended as context gates, not standalone signals.

Fixed experiment
----------------
One depth4 XGBoost fit per held fold, 800 rounds and the incumbent parameters.
Each fit uses 400,000 original real rows plus exactly 200,000 positive and
200,000 hard-negative synthetic rows. Synthetic rows receive 25% of each
covered (time,target) cell's original mass; real rows retain the rest.

The candidate backbone is the exact float32 mixture 80% Model26 component and
20% new matched-context model, then the unchanged Model9 stack, Model11 blend,
and rolling128 postprocess. The weight and all data rules were fixed before
scores were observed.

Evidence and gates
------------------
Every new fit is out-of-fold on complete unseen series and also excludes the
adjacent stopping fold. The feature selection and outer ensemble are inherited
and development-informed, so this is not a fully virgin pipeline. The reserved
outer holdout and all test labels remain closed unless the OOF target is met.

Target: held-series OOF TS-AUC >=0.640. Promotion requires gain >=0.0007 over
Model26, at least four of five fold wins, paired whole-series bootstrap lower
95% bound >0 (1,000 draws, seed271109), and positive gains in both prespecified
disjoint series-ID halves. No automatic deployment.

Research basis
--------------
Adams & MacKay (2007), Bayesian Online Changepoint Detection:
https://arxiv.org/abs/0710.3742
Puchkin et al. (AISTATS 2023), A Contrastive Approach to Online Change Point Detection:
https://proceedings.mlr.press/v206/puchkin23a.html
TS-CP2 contrastive predictive coding (ICLR 2022):
https://arxiv.org/abs/2011.14097
