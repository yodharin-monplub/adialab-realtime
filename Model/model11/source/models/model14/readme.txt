Model 14: time-varying pairwise stack

Extends Model 9's 34 primitive OOF margins with Model 11's new multiscale
probability and strongly regularized elapsed-time interactions.
CPU only. Run: OPENBLAS_NUM_THREADS=1 python3 models/model14/run_model14.py
Requires Model 11 to have completed all five folds first.

Pairs always compare positive and negative rows at the same online time.
Weights may change smoothly with elapsed time, which is available during live
inference. No final online length or future observation is used.

This is exploratory meta-level cross-validation, not fully nested base-model
validation. Upstream voters and model selection reused the development folds.
Outer-holdout and test labels are excluded. No automatic submission.
See README.md and runs/*/summary.json for the full protocol and measured results.
