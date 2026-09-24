Model31 — direct meta-stack for the matched backbone

Prespecified before observing any Model30 fold other than fold 0.

Motivation
----------
On pilot fold 0, Model30's raw component improved strongly over Model26's raw
component, while replacing the old stack column reduced the final score. This
isolates stack calibration, rather than signal quality, as the failure mode.

Frozen method
-------------
Fold 0 alone trains one L2-regularized logistic meta-model. Input columns are:

1. Model26's final ensemble margin.
2. Model30's raw component logit.
3. log(1 + online time).

Columns are standardized using fold-0 means and standard deviations. The fit
uses time/label-balanced weights, sklearn LogisticRegression with lbfgs,
C=0.01, tolerance 1e-8, at most 500 iterations, and seed 310911. Its probability
is the final output; the legacy stack is not applied a second time.

Folds 1-4 are untouched confirmation data for this exact method. Promotion
requires confirmation mean TSAUC >= 0.640, gain >= 0.001 versus Model26, at
least 3/4 fold wins, a positive whole-series bootstrap lower bound, and positive
gain in both fixed ID-hash halves. The five-fold mean is descriptive only.

CPU only. No test labels, upload, or cloud submission.
