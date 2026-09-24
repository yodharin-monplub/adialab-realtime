Model74: cross-fitted age-band selector over Models 39, 62, 63, 64 and 67.

The strongest frozen OOF predictions disagree by online age: Model63 is best
at very short ages, Model39 around ages 21-40, Model64 around ages 41-80,
and Model67 at long ages. A same-data time-adaptive blend would be optimistic,
so this experiment uses leave-one-fold-out selection. For every held-series
fold and each of eight prespecified age bands, it measures the five frozen
sources only on the other four folds. It switches away from Model67 only when
the best four-fold mean improves by at least 0.0005. The chosen source is then
applied to the fold whose labels were never used for that choice.

This is a standard cross-fitted meta-model: the reported prediction for every
row is out of sample for both its parent learner and its selector. It introduces
no new base-model fitting and never reads the sealed outer holdout or test
labels. Parent choices and the band boundaries were motivated by exploratory
development analysis, so the result remains exploratory rather than a virgin
holdout claim.

Promotion requires full held-series OOF TSAUC >=0.640, gain >=0.0005 over
Model67, at least 3/5 fold wins, positive paired-series bootstrap lower 95%
bound, and positive gains in two fixed series-ID halves.

Public evidence cautions that fitting blend weights on their scoring data can
create a false gain and that four-parameter time-adaptive blending regressed.
The disjoint-fold selector is designed to test that concern directly:
https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206

CPU only; no Vast.ai rental, packaging, upload, or cloud submission is automatic.
Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model74-mpl timeout 2h .venv/bin/python -u models/model74/run_model74.py
