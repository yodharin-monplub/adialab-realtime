Model116: learned causal temporal stacker over Submission30 score trajectories.

LightGBM (hard labels, time-pair weights, 1.5M rows, cross-fitted by held
fold) on per-row trajectory features of the chain and distilled z-scores:
EMAs (half-lives 2/8/32/128), running maxima (16/64/256/all), running mean,
16-step change, steps since max, t, plus five history descriptors.

Run: .venv\Scripts\python.exe -u models\model116\run_model116.py

Result: FAILED badly. Mean 0.61387 vs Submission30 0.63690 (-0.0230, 0/5).
Diagnosis (diagnose.py, fold 0; Submission30 = 0.656334):
  A  [t, zc, zd] only, weighted        0.650570 (-0.0058); gain zc 54% zd 37% t 9%
  B  full features, unweighted         0.626124 (-0.0302); history cols dominate
  C  full features, small leaves       0.633254 (-0.0231); history cols dominate
Cause: a tree model with max_bin=127 outputs a piecewise-constant function of
the quantized scores, so fine within-stratum ordering collapses into ties
(TS-AUC credits ties 0.5); the linear blend keeps the full continuous order.
With all features the constant-per-series history descriptors (cols 38-41)
absorb the splits and make it worse. Not a leak and not a code bug: tree
stackers over OOF scores are the wrong model class here; only smooth
(linear) stackers are a fair test. Submission30's fixed blend stands.

Linear follow-up (weighted logistic regression, fold 0, 600k rows):
  D  [t, zc, zd]                       0.656129 (-0.0002)  = reproduces the fixed blend
  E  all 43 trajectory features        0.650016 (-0.0063)
So the trajectory features hurt even in a smooth model: learned temporal
stacking over these OOF scores is closed. Final: keep the fixed linear blend.
