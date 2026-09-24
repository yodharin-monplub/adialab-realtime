Model73: deployable direct Model62 LightGBM voter on the Model26 stack.

The frozen Model62 fold learners replace Model26's paired-XGBoost core voter.
The 31 detector-bank voters, other two feature components, stack coefficients,
rolling-max correction and compact-model blend are unchanged. This deliberately
does not include Model62's research-only Model35 anti-residual or Model38 EMA
corrections, so its score must not be confused with Model62's 0.637534 OOF.

On the reused five held-series development folds, this direct-voter variant
scores 0.6354292998339404 versus Model26's 0.6324069786337266; four of five
folds improve, but fold 3 regresses. Confirmation folds 2-4 improve by only
0.0005400 on average. The result is exploratory and not an independent
holdout or a cloud-score prediction. No fitting uses test labels.

The user's requested next step is one cloud probe before more modeling. A
separate submission bundle must pass resource hashing, offline/streaming parity,
determinism, Python 3.12 compatibility and runtime/RAM checks before upload.
Cloud training is disabled; inference is CPU-only.
