Model34 — TSAUC-aligned residual boosting over Model26

Model33 showed that replacing Model26 with a row-wise nonlinear stack destroys
useful same-time ranking. Model34 instead freezes Model26 as the initial margin
and lets a small LightGBM learn an additive correction using an exact
same-time positive/negative pairwise logistic objective.

Inputs are the Model33 level-one columns except Model26's own margin: 34 frozen
primitive logits, four distribution summaries, raw Model27 and Model30 logits,
log online time, and 13 clean-history contexts (54 columns). Each held fold and
its adjacent stopping fold are excluded. The fixed recipe uses 400,000 real
rows, 500,000 TSAUC-weighted pairs, 120 trees, seven leaves, learning rate .01,
minimum leaf size 2,000 and L2 200 on four CPU threads.

Promotion requires mean held-series OOF TSAUC >= .640, gain >= .0007 versus
Model26, at least 4/5 fold wins, positive whole-series bootstrap lower bound,
and positive gains in both fixed ID-hash halves. No cloud submission is made
automatically. The inherited Model9 primitive nesting caveat remains.
