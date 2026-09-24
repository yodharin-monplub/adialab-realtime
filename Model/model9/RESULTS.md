# Model 9 Results

Run: `models/model9/runs/fulldata_819a001db6/`  
Completed: 2026-09-03

## Outcome

Model 9 did not reach the 0.64 TSAUC target and does not replace Model 6 yet.
The outer holdout was not read. All numbers below use the 8,333-series
development population; the 1,667-series outer holdout stayed sealed.

The strongest Model 9 result is the exploratory TSAUC-aligned pairwise stack:

| Result | Mean TSAUC | Fold SE | Comparison |
| --- | ---: | ---: | --- |
| Model 6 frozen development baseline | 0.625675 | 0.008440 | reference |
| Model 9 pairwise stack | **0.628020** | 0.007603 | +0.002345 +/- 0.001694 paired SE; wins 4/5 folds |
| Model 9 adaptive baseline blend | 0.625205 | 0.008419 | -0.000470 +/- 0.000496 paired SE |

The pairwise result is level-two OOF screening, not confirmatory validation.
Some base voters used to form a meta-training row can have trained or
early-stopped on the meta-held fold, and Model 5 selected detector banks using
these development folds. A fully nested base refit is required before using it
in a submission.

## Fixed LightGBM scale study

The study completed 40 fits: four fixed architectures, two nested row caps,
and five complete-series-disjoint score folds.

| Architecture | Fit rows | Raw mean TSAUC | Fold SE |
| --- | ---: | ---: | ---: |
| Joint Fisher-768 | 1,200,000 | **0.615103** | 0.007368 |
| Joint Fisher-768 | 400,000 | 0.614726 | 0.007061 |
| Candidate split-512 | 1,200,000 | 0.613853 | 0.007769 |
| Candidate all-1,415 | 400,000 | 0.613027 | 0.008291 |
| Candidate all-1,415 | 1,200,000 | 0.612354 | 0.007730 |
| Candidate split-512 | 400,000 | 0.610801 | 0.007940 |
| Core Fisher-512 | 1,200,000 | 0.609263 | 0.007318 |
| Core Fisher-512 | 400,000 | 0.608847 | 0.007784 |

The joint model's 1.2M-minus-400k paired gain was only 0.000378 +/- 0.002017
(2/5 fold wins). The clearest scale result was candidate split-512:
1.2M rows improved on 400k by 0.003052 +/- 0.002178 and beat candidate-all at
1.2M by 0.001499 +/- 0.000406 on all five folds. This supports feature
selection before scaling, but a single LightGBM remains weaker than Model 6's
heterogeneous ensemble.

Postprocessed, selected, and blended scale results are exploratory and are not
used to claim improvement. Promotion and meta-stack stages were not run because
the scale study did not justify their cost or adaptive-validation risk.

## Decision

Keep Model 6 as the submission baseline. The next defensible Model 9 step is a
fully nested refit of the pairwise stack's 34 primitive inputs, followed by a
frozen inference implementation and evaluation on genuinely untouched data.
