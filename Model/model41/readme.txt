Model41: early-only anti-residual correction with confirmed smoothing.

Fourteen simple time gates were screened on pilot folds 0 and 1 only. The
largest pilot robustness score came from applying Model35's anti-residual
delta while time_online < 128, then reverting to Model26. Model38's already
confirmed half-life-4, weight -0.20 smoother is applied afterward.

Folds 2, 3, and 4 were excluded from gate selection and are the confirmation
set. CPU only. No training, packaging, upload, or cloud submission.

