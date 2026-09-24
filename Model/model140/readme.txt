Model140 - STAGE 4 of the user's protocol: from the statistical tests to features, and selection.

fast_tests.py: numba re-implementations of the stage-3 winners (identity
ranking) computed causally for ALL 4.2M real rows and the synthetic rows:
prefix scans (CUSUM of squares, variance LR scan, Chow-F scan, Buishand range,
Page-Hinkley), window-128 vs history (Levene-type mean-|dev| ratio, Brown-
Forsythe, |log variance ratio|, Wasserstein-1, Kuiper, Ljung-Box(10),
kurtosis difference) and 64-point spectral entropy difference = 13 columns.

run_model140.py (to write when stage 3 finishes): same student, feature sets
  A  current 317 columns (control)
  B  317 + 13 test features (no selection)
  C  317 + tests significant at p<.01 / .05 / .10 only
  D  LightGBM gain top-k of (317 + 13), k in {100, 200}
  E  correlation pruning |rho| > 0.95 of (317 + 13)
  F  forward addition of the 13 one at a time by pilot-fold gain
Pilot folds 0-1 -> folds 2-4 for the best -> holdout.
Status: builder fixed (acf lags capped at n-2) and run 09:11-09:20: real.npy
(4,195,720 x 13, all finite) + syn_fold0-4.npy. Single-column TS-AUC on all
dev rows matches the stage-3 ranking: cusum_sq .551, wasserstein .552,
var_lr_scan .552, kuiper .545, |log var ratio| .545, levene .539,
ljung_box .477 (inverted), kurtosis .514, spec_entropy .513.
run_model140.py written 09:08; pilot (folds 0-1) started 10:06 with 6 threads.
Arm C is degenerate: all 13 features pass p<.01 (largest p = kurtosis .007),
so C01 = C05 = C10 = B; the p-value selection already happened in stage 3
(130 -> these 13). Pilot arms therefore B, D100, D200, E; F last.
Pilot result (gain vs A, fold 0 / fold 1; done 16:06 after a restart alone):
  B    317+13           -0.0009 / +0.0031   mean +0.0011
  D100 gain top-100     -0.0004 / +0.0020   mean +0.0008  (fits 2.5x faster)
  D200 gain top-200     -0.0001 / -0.0002   mean -0.0002
  E    |rho|>.95 pruned +0.0009 / +0.0021   mean +0.0015  (220 / 196 columns)
Five-fold result 16:34 (gain vs A per fold, score.txt):
  B  mean +0.0017 s.e. 0.0015 wins 3/5  -0.0009 +0.0031 +0.0059 -0.0025 +0.0026
  E  mean +0.0030 s.e. 0.0009 wins 5/5  +0.0009 +0.0021 +0.0034 +0.0021 +0.0064
  P  mean -0.0005 s.e. 0.0009 wins 2/5  -0.0029 +0.0020 -0.0007 -0.0021 +0.0013
E passes the stage rule (3.3 s.e., 5/5) - the first arm of the protocol to do
so. Control P (the same |rho|>.95 pruning on the 317 alone, 172-221 columns)
is flat, so the gain is the 13 test features, usable once their near-
duplicates among the 317 are removed (B, without pruning, only +0.0017 3/5).
Dev-level blend with the Submission28 chain (select_blend.py, constants_e140):
E voters +0.0027 at w=.5 vs a050's +0.0015 -> +0.0012 over Sub30 on dev.
Serving: fast_online.FastTests (numpy, per-step, parity 0.0 vs the numba
batch on 12 series; 1.4 ms/step). holdout_eval140.py = Sub31 with the a050
voters swapped for the E voters.
HOLDOUT (1,667 sealed series, 840,797 rows, 17:07-19:12):
  Sub28 chain 0.62969  Sub30 0.63064  Sub31 0.63222  candidate 0.63245
  gain vs Sub31 +0.00023 (holdout noise +/-0.002) -> does not transfer.
Same pattern as every dev-fold gain below ~0.005 in this project: the +0.0030
raw / +0.0012 blended dev gain becomes +0.0002 on the sealed series. Not a
submission candidate (holdout_results_E.json, holdout_predictions_E.npz).

Arm ED (20:45, after the stage-2 decision): the 13 tests also on the
differenced series (FT_TRANSFORM=diff -> data/real_diff.npy, syn_diff_*) and
on |x - history mean| (FT_TRANSFORM=abs), 317 + 39 columns, same pruning
(FT_SETS=identity,diff,abs). Single-column TS-AUC on all 4.2M dev rows
confirms stage 3: diff lifts the scale tests (cusum_sq .551 -> .562,
var LR .552 -> .558, |log var ratio| .545 -> .553, Wasserstein .552 -> .558),
abs lifts the location scans (Buishand .530 -> .560, Chow .530 -> .557).
ED result (01:15-01:36, 194-250 columns kept):
  ED mean +0.0026 s.e. 0.0019 wins 3/5  -0.0010 -0.0004 +0.0056 +0.0004 +0.0086
Not better than E (+0.0030, 5/5) and fails the rule: the diff / abs versions
of the tests add nothing the identity versions + the 317 do not already
carry. Stage 4 closed; E remains the dev-only winner (holdout +0.0002).
