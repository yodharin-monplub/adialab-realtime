Model144 - the user's idea (2026-09-20): two-sample tests between the PRE-break
and POST-break segments as features, with the break time ESTIMATED (reading C,
confirmed by the user: "comfirm c try it").

Why it is different from what existed: stage 3 / the 4,741 detectors compare
the history with "everything online so far" (or a fixed trailing window), so a
recent break is diluted by the clean online points before it. Here, at every
step t, split_tests.py
  1. scans every candidate split k of the online prefix with the Gaussian
     mean+variance likelihood ratio (pre = history tail + o[0..k), post =
     o[k..t], >= 8 points) -> estimated split k_hat, the LLR at k_hat, the LLR
     of "whole online vs history" (k = 0), sharpness (max - mean over k), the
     relative position, the post length and how many steps k_hat has been
     stable ("age");
  2. runs a battery between the clean reference and the estimated post segment
     o[k_hat..t] (last 512 points at most): Welch t (signed, |.|), log variance
     ratio (signed, |.|), KS and Kuiper (scaled by sqrt(nm/(n+m))), Wasserstein-1,
     Mann-Whitney |z|, mean-absolute-deviation ratio, lag-1 autocorrelation
     difference, kurtosis difference, roughness ratio;
  3. repeats the scan on the first-differenced series (the stage-2 winner) with
     a smaller battery, plus the disagreement of the two split estimates.
27 columns (NAMES), all scale-free (series z-scored by its own history).

LEAKAGE RULES (agreed with the user): tau is never read by the feature code;
row t uses only the history and o[0..t]; all constants come from the series'
own history; test selection on training folds only; sealed holdout once.
row_features() computes ONE step from the prefix plus a 4-number carried state,
and the batch builder is a loop over it, so serving is the same code.
`split_tests.py check`: features of a truncated series == the full series'
rows, and perturbing the future leaves earlier rows bit-identical - PASSED
2026-09-20 on 6 series / 2,923 steps before any training.

run_model144.py: same Model111 student, folds, weights, splices as stage 4.
  B  317 + 27 (no selection)      E  317 + 27, |rho| > .95 pruned
  X  317 + 13 (Model140 tests) + 27, pruned
Rule: gain > 2 s.e. and >= 4/5 folds, then the sealed holdout.

RESULTS 2026-09-20 (gain vs the current student, per fold; score.txt)
Single column, all 4.2M dev rows: best = spd_llr_max 0.563 (LLR scan on the
differenced series; the best stage-3 test was 0.565), sp_llr_max 0.549,
spd_llr_sharp 0.557, sp_rough_ratio 0.546, sp_logvr 0.543, sp_kuiper 0.540,
sp_ks 0.524 - the battery at the estimated split is individually WEAKER than
the same tests on a fixed 128-window (0.545-0.551).
  E  mean +0.0011 s.e. 0.0008 wins 3/5  +0.0033 -0.0007 -0.0001 +0.0023 +0.0006
  X  pilot only                          -0.0005 -0.0015
-> does not pass the rule. By online time the gain is flat (fold 0: +0.002 to
+0.005 in every bucket; fold 1: negative in every bucket below 512), i.e. no
early-detection benefit - it behaves like "more features of the same kind".

WHY (diagnose_split.py, uses tau for diagnosis only): the split estimate is
poor. Median |k_hat - tau| is 25-50 steps 32-64 points after the break and
grows to 126-140 steps after 256 points; only 10-14% of series have k_hat
within 10 steps of the truth. Most breaks in this data are not visible as a
Gaussian mean/variance shift even with hundreds of post-break points, so the
battery runs on a wrongly placed segment, and under no break the argmax picks
the most unusual-looking tail (selection effect).
WHITENED VARIANT (SP_WHITEN=1 -> data_white/, runs_white/): the same scan +
battery on the one-step-ahead residuals of an AR(p) fitted on the history only
(models/model57/whitening.py, BIC over p in 0,1,2,4,8,12). Leak checks passed.
Split accuracy improves only slightly (within 10 steps after 256 points: 13%
vs 10%), but the features become much more useful to the model:
  E-white  mean +0.0037 s.e. 0.0013 wins 5/5  +0.0061 +0.0017 +0.0038 +0.0001 +0.0068
-> PASSES the rule; best development arm of the whole protocol.
Dev blend with the chain (select_blend, constants_e144.json): +0.0028 at w=.5
(current student +0.0015, stage-4 E +0.0027 - and that one gave only +0.0002
on the holdout, so the holdout decides).
Serving: StreamingSplitTests (same row_features as training); `parity`:
streaming vs batch worst relative difference 0.0 on 8 series / 4,847 steps.
holdout_eval144.py = Submission31 with its distilled voters replaced by the
E-white voters (resumable, checkpoint every 100 series; the first run was
killed by the low-memory reaper at 700/1667).
SEALED HOLDOUT (1,667 series, 840,797 rows, 8,003 s with 6 workers):
  Sub28 chain 0.62969 | Sub30 0.63064 | Sub31 0.63222 | candidate 0.63274
  gain vs Sub31 +0.0005  (stage-4 E was +0.0002; holdout noise +/-0.002)
-> positive but not distinguishable from noise and below the +0.001 bar:
NOT a submission candidate on its own. Kept as a component for a combined
candidate (with external data / ranker if those transfer).
