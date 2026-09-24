Model123: outer-holdout evaluation of Submission28 vs Submission30.

Purpose
The 1,667 series with inner_series_fold == -1 were sealed by Model5 and never
used for fitting or selection in any model, including the Model109 teachers
(their privileged features exist only for the 8,333 development series). The
sealed Submission30 bundle (submissions/submission30) is run on these series
exactly as the cloud runs it, with 8 worker processes; every online step
records both the Submission28 chain score and the Submission30 blended score.
This gives (a) an independent Sub28 vs Sub30 comparison on untouched labeled
data and (b) a cleaner local reference for the local->cloud offset.

Labels: y_t = 1 for t >= tau_index (online index), else 0; break-free series
are all zero (tau_index = -1).

Run: .venv\Scripts\python.exe -u models\model123\holdout_eval.py --workers 8
Outputs: holdout_predictions.npz, holdout_results.json (TS-AUC for both).
Result (1,667 series, 840,797 rows, 5,453 s with 8 workers)
  Submission28 chain   0.629689
  Submission30 final   0.630638   (+0.00095)
  ID halves: +0.0043 / -0.0024 (fragile); by online time: Sub30 loses at
  t<32 (-0.0065, tiny weight) and t>=512 (-0.0030), gains at 64-256 (+0.004).

KEY FINDING: cloud public score of Submission30 = 0.629568, i.e. the sealed
holdout (0.6306) matches the cloud within 0.001, while the reused five
development folds gave 0.6369. The ~-0.007 "local->cloud offset" seen since
Model26 is optimism from adaptive selection on the development folds (100+
models chosen against them), NOT a scoring or distribution difference.
Consequences: (1) development-fold gains below ~0.005 are not trustworthy;
(2) the sealed holdout is now the go/no-go reference for future bundles
(cost: ~1.5 h per bundle with 8 workers); (3) it has now been used once for
Sub28 vs Sub30 only, no parameter was chosen on it.

Scorer check: the official scoring.py loop (per online step, sklearn
roc_auc_score, n_pos*n_neg weights) and tsauc.py agree to 12 decimals on all
840,797 holdout rows, also for heavily tied predictions.

Pre-registered confirmation of the blend weight (selection on dev folds,
confirmation on holdout; the distilled z-score is recovered exactly from the
stored chain and final scores):
  chain only 0.629689 | w=0.30 0.630800 | w=0.35 0.630794 | w=0.50 (deployed)
  0.630638 | w=0.75 0.630110 | w=0.35 with no distilled term for t<32 0.630891
The dev-all-folds optimum (0.3-0.4) beats the deployed 0.5 by +0.0002 (+0.0003
with the t<32 gate): inside the +/-0.003 half-split noise, not worth a second
cloud run. No parameter was changed as a result.
