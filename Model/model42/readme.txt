Model42: pilot-selected secondary smoothing of Model39.

The frozen Model10 causal-transform bank was screened on Model39 using folds 0
and 1 only. Among transforms that improved both pilot folds, the strongest
robust candidate was a logit EMA trend with half-life 8 and correction weight
-0.10. Model42 applies that exact transform once to Model39.

Folds 2, 3, and 4 are the confirmation set and did not choose the transform.
CPU only. No training, packaging, upload, or cloud submission.

