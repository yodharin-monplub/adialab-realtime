Model109: oracle-distilled LightGBM student ("learning with privileged information").

Motivation
The public leaderboard leader (0.6630, 2026-09-16) is a team called
"distilled-oracle". Our own analysis showed the breaks are tiny: even with the
true break time and >=100 post-break points, a crude offline series classifier
reaches only AUC ~0.72, and hard 0/1 labels at early post-break ages are nearly
pure noise. Hard-label training therefore has high variance.

Idea
A teacher that sees the *whole* online segment (future included) is trained
with cross-fitting. Its probabilities are used as soft targets for a causal
student that only sees the usual 500 causal features (Model20 439-column cache
plus Model57 61 whitening columns). By the tower property
E[P(y_t | full series) | past] = P(y_t | past), so the student still targets
the correct causal posterior, but with lower-variance targets.

Privileged features (teacher_features.py, never deployable)
History-fit AR(10) innovations; for every candidate onset k the Gaussian
mean/scale LLR of innovations[k:] and raw[k:] and a lag-one dependence LLR;
argmax onset, maximum evidence, uniform-prior posterior CDF of the onset at t,
age since the estimated onset, prefix LLR of the estimated post segment,
post-segment moments, online length L and t/L, and history summaries.

Protocol (run_model109.py)
Original Model5 development folds (8,333 series), outer holdout sealed.
For held fold k, four teachers are each trained on the three folds excluding k
and j and predict fold-j rows of the student sample: fold-k labels never reach
any model predicting fold k. Students (500k rows, time-pair weights, Model84
expressive LightGBM settings, 8 threads) are trained with targets
  hard = y,  mix = 0.5*y + 0.5*teacher,  soft = teacher
on identical rows. Raw held-fold TS-AUC is reported (no stack/postprocess).
Reference: Model84 expressive raw components average ~0.6195 per fold
(fold0 0.6433, fold1 0.5996) with 400k real + 400k synthetic rows.

Run from project root (Windows):
  .venv\Scripts\python.exe models\model109\teacher_features.py
  .venv\Scripts\python.exe -u models\model109\run_model109.py 0 1

No upload or submission performed.
