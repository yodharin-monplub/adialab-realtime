Model32 — five-fold cross-fitted diversity stack

Prespecified before observing Model30 folds 3 and 4.

For each held fold f, train a separate L2-regularized logistic meta-model on
the other four folds' already-OOF level-one predictions. Input columns are:

1. Model26 final ensemble margin.
2. Model27 matched-context raw logit.
3. Model30 matched/no-context raw logit.
4. log(1 + online time).

Each meta model standardizes columns on its four training folds, uses the same
time/label-balanced weighting as the backbone studies, and uses sklearn lbfgs
LogisticRegression with C=0.01, tolerance 1e-8, at most 500 iterations, and
seed 320911+fold. The held fold is never used for fitting or standardization.
The meta probability is the final output; there is no second legacy stack.

Promotion requires five-fold held-series OOF TSAUC >= 0.640, gain >= 0.0007
versus Model26, at least 4/5 fold wins, a positive whole-series bootstrap lower
bound, and positive gain in both fixed ID-hash halves.

This is a fixed, cross-fitted stacked-generalization test. CPU only. No test
labels, upload, or cloud submission.
