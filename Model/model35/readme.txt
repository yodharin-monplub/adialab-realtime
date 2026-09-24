Model35 — exploratory anti-residual post-processing

Model34's TSAUC-aligned correction was directionally redundant with Model26.
An explicitly exploratory grid diagnostic found that subtracting 0.75 times
that correction and 0.10 times Model30's matched-synthetic component delta
gave the best mean development OOF score among the inspected coarse grid.

This model performs no new fit. It records the fixed post-process, its five
fold scores, fixed ID-half gains, and a 1,000-replicate whole-series bootstrap.
Because the coefficients were selected after observing all five development
folds, its result is adaptive evidence—not untouched confirmation. Promotion
still requires TSAUC >= .640, gain >= .0007, 4/5 wins, positive bootstrap lower
bound and positive gains in both ID halves. No automatic cloud submission.
