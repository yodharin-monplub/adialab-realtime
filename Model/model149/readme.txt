Model149 - break EVIDENCE integrated over the unknown break position
(second generation of the user's split-test idea, 2026-09-22).

Why: Model144 tests the series at the single most likely split, but the split
estimate is poor (only 10-13% of series within 10 steps of the truth even 256
points after the break). A participant on Discord reported that knowing tau is
worth +0.019 and that INTEGRATING over the unknown tau recovered part of it.
evidence.py: for 5 channels of the AR-whitened, history-standardized series -
level z, first difference, |z| (scale), z_t*z_{t-1} (dependence), rank-Gauss
of z (shape-free) - the Gaussian mean+variance likelihood ratio of "break at
k" is computed for ALL k and summarised as: max (= Model144's statistic), log
marginal evidence under a uniform prior on k, posterior mean and sd of the
position, posterior mass of "break within the last 16 / 64 steps"
= 30 causal columns. Leak checks (prefix invariance, future perturbation)
passed before any training.
run_model149.py: exactly the Submission33 student recipe (splices + one
re-cut set at 40%, soft targets) with columns 317 + 27 split + 30 evidence,
|rho| > .95 pruned per fold; arms EV40 / EV40b / EV40c mirror SRE40 / b / c of
Model147, so the bag of the three vs BAG3 changes one thing only.
RESULTS 2026-09-22
Single column TS-AUC on all 4.2M dev rows: the log marginal evidence beats the
maximum in every channel - z .580/.576, dz .584/.581, |z| .588/.586, lag .575/
.574, rank-Gauss .585/.579; posterior sd .43 (i.e. .57 flipped), recent mass
r16/r64 .43-.46. ev_abs_lme = 0.588 is the best single column ever measured
here (best stage-3 test 0.565, Model144's sp_llr_max 0.549).
In the student (same data as the Model147 students, 374 -> ~250 columns kept):
  EV40  vs SRE40   +0.0031 -0.0069 +0.0032 -0.0019 -0.0021   mean -0.0009
  EV40b vs SRE40b  -0.0012 -0.0021 +0.0049 +0.0003 -0.0049   mean -0.0006
-> no gain (10 fold comparisons, mean -0.0008, 4 wins): the evidence columns
are individually strong but carry nothing the 317 + 27 columns do not already
give the model. EV40c stopped, no bag, no holdout. Closed.

Decision rule (not reached): bag dev gain vs BAG3 must be clearly positive (>= +0.002,
>= 4/5) before spending a 3-hour holdout run; lesson of Model148: +0.0013 on
dev (5/5) was -0.0009 on the holdout.
