Model139 - STAGE 3 of the user's protocol: one statistical test at a time.

Each test is ONE causal feature: the test statistic of the online window (last
32 / last 128 / whole prefix) against the history (last 1,024 points), or the
window's own statistic minus the history's (dependence/shape tests), or a
change-point statistic on the whole prefix, computed at 150,000 sampled online
rows (same rows for every test and transform). stat_tests.py holds 48 tests:
  location  welch_t student_t mann_whitney brunner_munzel mood_median sign_test
  scale     log_var_ratio levene brown_forsythe ansari_bradley mood_scale fligner mad_ratio
  distrib.  ks anderson_k cramer_vm kuiper epps_singleton chi2_cells8 energy wasserstein
  shape     jarque_bera skew_z kurt_z shapiro_w lilliefors
  depend.   ac1 ac2 ac3 ljung_box10 box_pierce10 durbin_watson runs_z turning_points
            variance_ratio4 arch_lm abs_ac1
  spectral  spec_low spec_high spec_entropy spec_peak
  change-pt cusum_mean cusum_sq page_hinkley chow_f_scan var_lr_scan pettitt buishand_range
and 8 transforms (stage 2): identity, abs, square, sign*sqrt, sign*log1p,
diff, rank-to-Gaussian, rank-to-Laplace (rank maps use the history ECDF, causal).

evaluate.py: per (transform, test, window) TS-AUC, series-bootstrap s.e. (100
resamples), z = (TS-AUC - 0.5)/s.e., two-sided normal p (a feature below 0.5
is usable with its sign flipped; the signed TS-AUC is reported), flags at
p < .01 / .05 / .10. Output data/results_<transform>.csv.

Run: python models/model139/stat_tests.py build identity 8 ; python models/model139/evaluate.py identity
Results, identity transform (60,000 rows, 05:19): 130 non-degenerate (test,
window) features; significant at p<.01/.05/.10 = 79/84/86 (with |statistic|
variants for sign-dependent tests; 70/74/77 signed only). Best window per
family (TS-AUC +/- s.e.):
  change-point  CUSUM-of-squares 0.556+/-.006, variance LR scan 0.554, Buishand
                0.534, Chow-F scan 0.534, Page-Hinkley 0.525, CUSUM-mean 0.519
  scale         Levene(w128) 0.550, Brown-Forsythe 0.550, Fligner 0.548,
                |log var ratio|(w128) 0.545, |MAD ratio| 0.536, Mood 0.527
  distribution  Wasserstein(w128) 0.551, energy 0.545, Kuiper 0.546, Anderson
                0.545, Epps-Singleton 0.541, chi2 cells 0.539, CvM 0.537, KS 0.537
  shape         Shapiro W 0.470 (i.e. 0.530 flipped), kurtosis 0.529, JB 0.527
  dependence    Ljung-Box 0.467 (0.533 flipped), Box-Pierce 0.468, |ac1|(w32)
                0.525, |Durbin-Watson| 0.525, |ac2| 0.519
  spectral      spectral entropy 0.534, |high band| 0.518, low band 0.483
  location      ALL n.s. signed (Welch/Student t, Mann-Whitney, Brunner-Munzel,
                Mood median, sign test): breaks shift the mean both ways.
Window 128 > 32 nearly everywhere; whole-prefix change-point scans strongest.
Full grid (8 transforms, 20:20; summary.py -> data/summary.csv):
  transform    sig .01/.05/.10   best test        avg of top-20 identity tests
  identity      79/ 84/ 86       cusum_sq 0.556   0.547
  abs           88/ 94/103       buishand 0.563   0.546
  square        82/ 90/ 97       buishand 0.565   0.547
  sqrt          71/ 79/ 86       cusum_sq 0.560   0.542
  log           67/ 78/ 82       cusum_sq 0.561   0.545
  diff          71/ 75/ 82       cusum_sq 0.564   0.555
  rankgauss     74/ 85/ 90       cusum_sq 0.561   0.546
  ranklaplace   74/ 83/ 86       cusum_sq 0.559   0.548
diff is the best transform for the scale/distribution/change-point tests
(+0.008 on the top-20 average, best for 12 of the top 25 families); abs and
square wake up the location scans (Buishand 0.565, Chow 0.562, CUSUM-mean
0.559) because a variance change becomes a mean change; sqrt/log/rank
transforms change nothing (within 1 s.e.); 46 combinations are significant
at .01 under every transform. Best single test anywhere: 0.565.
