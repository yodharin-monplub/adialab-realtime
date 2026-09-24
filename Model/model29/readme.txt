Model29 — pilot-trained context reliability gate

Model27 improved fold 0 but hurt fold 1, invalidating a single aggressive blend.
Model29 freezes a regime-aware linear gate after observing only those two pilot
folds. The gate uses Model26 and Model27 logits, their disagreement, online time,
13 clean-history descriptors, and disagreement-by-context interactions.

The logistic gate is trained with time/label-balanced weights on folds 0-1 only.
Folds 2-4 are untouched confirmation data. Success requires confirmation TSAUC
>= 0.640, gain >= 0.001 versus Model26, at least 2/3 wins, a positive whole-series
bootstrap lower bound, and positive gain in both fixed ID-hash halves.

CPU only. No test labels, packaging, upload, or cloud submission.
