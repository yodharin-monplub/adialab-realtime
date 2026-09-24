Model39: Model35 anti-residual margin plus Model38's confirmed EMA smoother.

Model38 selected its half-life-4, weight -0.20 logit smoother on folds 0-1 and
then improved Model26 on all three untouched confirmation folds. Model39
transfers that exact frozen transform, without a new grid or parameter search,
to Model35's stronger anti-residual margin.

Model35 is exploratory because its anti-residual coefficients were selected
adaptively on reused development OOF. Model39 therefore remains exploratory
even if the transferred smoother is stable. It is compared with both Model26
and unsmoothed Model35 by whole-series bootstrap and fixed ID-hash halves.

CPU only. No training, packaging, upload, or cloud submission.

