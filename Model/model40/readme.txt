Model40: two-pass short-memory smoothing of Model35.

Model39 applies Model38's independently selected and confirmed half-life-4,
weight -0.20 logit smoother once to Model35. A diagnostic showed that applying
the identical causal transform a second time improved all five folds, while a
third pass reduced the mean. Model40 freezes exactly two passes.

The number of passes was inspected on reused development OOF, so Model40 is
explicitly adaptive/exploratory even if its paired bootstrap is positive.

CPU only. No training, packaging, upload, or cloud submission.

