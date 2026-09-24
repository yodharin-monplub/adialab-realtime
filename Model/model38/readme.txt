Model38: pilot-selected short-memory smoothing of Model26.

The complete frozen Model10 causal-transform grid was screened on development
folds 0 and 1 only. The candidate with the largest positive robustness score
among transforms that improved both pilot folds was:

    logit_ema_trend, half_life=4, correction_weight=-0.20

This subtracts 20% of the current logit's deviation from its inclusive
four-step EMA, suppressing isolated score spikes while preserving most of the
current Model26 margin. Folds 2, 3, and 4 are reserved for confirmation and
play no role in choosing the transform.

CPU only. No training, packaging, upload, or cloud submission.

