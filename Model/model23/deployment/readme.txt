Submission23 — within-fold equal-probability XGBoost mixture
=========================================================
Pretrained CPU inference only. forceFirstTrain=false, trainFrequency=0,
INFER_PARALLELISM=12; numerical libraries use one thread per worker.
Target: 16vCPU/64GiB. No training data, labels, donors or credentials bundled.

Only the core XGBoost component changes from Submission20: each of its five
fold voters becomes a pair (Model20 aligned25 and Model21 aligned60).
First average the two raw probabilities in float32 within each fold; then
apply the unchanged five-fold logit bag. Do NOT average all ten logits, or
average two already-bagged probabilities: neither matches the tested recipe.
All other model resources, detectors, linear stack coefficients, causal
rolling128 correction, and compact components remain unchanged.

Local full-ensemble mean-fold TS-AUC0.632094726 versus Model20's0.631575537.
Gain+0.000519188, wins4/5, simultaneous95% lower gain+0.000091884.
These are exploratory reused-development results, not independent or cloud
confirmation. Deployment combines five paired voters while local OOF scores
use the held-fold pair. Serving parity and runtime must be checked separately.
This bundle does not upload or start any cloud run itself.
