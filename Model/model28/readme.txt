Model28 — pilot-selected confirmatory blend

Purpose
-------
Test whether Model27's family-matched context model deserves substantially more
weight than the conservative 20% fixed in Model27.

Protocol
--------
Model27 fold 0 was inspected once as a pilot. Its full stacked TSAUC increased
monotonically through an 80% context-model weight, so Model28 freezes exactly
80% context model plus 20% Model26 before any fold 1-4 scores are observed.

Folds 1, 2, 3 and 4 are the confirmation set for the blend decision. The target
is confirmation mean TSAUC >= 0.640. Promotion additionally requires gain at
least 0.001 versus Model26, at least 3/4 fold wins, a positive whole-series
bootstrap lower bound, and positive gain in both prespecified ID-hash halves.

The five-fold score is descriptive only because fold 0 chose the weight. Model27
owns the trained XGBoost artifacts and its donor-exclusion contract. Model28 is
a frozen inference blend and does not retrain or touch test labels.

CPU only. No submission is automatic.
