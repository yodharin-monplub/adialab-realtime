Model86: independent-seed bag of Model84's expressive LightGBM voter.

Model84 improved full held-series OOF from 0.637609 to 0.637888, but one pilot
fold and one confirmation ID half regressed. Model86 holds its 63-leaf recipe,
400,000 real rows, 400,000 aligned synthetic rows, 40% synthetic cell mass,
317 features, exclusions, and postprocessing fixed. It changes only explicit
LightGBM bagging, feature-fraction, and histogram-construction seeds. Those
seeds independently control the stochastic row and column samples.

The new and Model84 expressive probabilities are averaged before the frozen
stack and causal transform. Pilot folds 0-1 select a fixed 0.375, 0.50, or
0.625 weight for this seed-bag signal against Model67; the corresponding
Model84 baseline uses 0.50 of the original expressive signal. A +0.0001 pilot
gain over Model84 is required before folds 2-4 train. Final promotion requires
full and confirmation TSAUC >=0.640 plus the standard robustness gates.

CPU-only and fold-checkpointed. No cloud run, packaging, or submission.
