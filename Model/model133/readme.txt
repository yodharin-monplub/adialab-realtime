Model133: causal "moments & transforms" feature bank (user suggestion, 2026-09-17 23:40).

The user suggested value transformations (absolute, sqrt, log/Laplace, Fourier)
and higher/central moments. Inventory of the 4,741 detector + 61 whitening
columns: absolute values (1,086), mean-square (68), empirical-CDF/quantile
cells (1,717), Haar wavelets (70), CUSUM/EWMA/autocorrelation banks exist;
MISSING were sqrt/log transforms, Fourier/spectral features, and the 3rd/4th/
5th moments of the online windows (kurtosis exists only for the history).

moment_features.py (numba, causal, per online step):
  windows 16 / 64 / 256 / whole prefix x {skewness, excess kurtosis, 5th
  standardized central moment, mean sqrt|x|, mean log1p|x|} as differences
  from the history's values (20 columns) + FFT band power fractions of the
  last 64 points (low 1-4, mid 5-16, high 17-32 bins) minus the history's
  (3 columns) = 23 columns; computed for the 4.2M real rows and the kept
  synthetic rows of each fold plan.
run_model133.py: Model111 recipe + 23 columns; pilot folds 0-1, then folds
2-4 if the pilot gains >= +0.002 on both.

Features built 23:46-23:55 (real 4.2M rows in 4 min; synthetic 30 s/fold).
Pilot (raw held-fold TS-AUC of the distilled student)
fold 0: 0.643142 vs Model111 0.644007 (-0.0009); new-column gain share 8.0%
        (top: prefix-window m4, m5, m3; 256-window m4, m5)
fold 1: 0.606563 vs 0.607235 (-0.0007); gain share 8.5%
Result: the tree uses the new columns but they add nothing on held data -
the moment/transform information is already carried by the empirical-CDF /
quantile cells (1,717 columns) and the |x| / x^2 banks. Closed 00:05.
Standalone TS-AUC of the new columns (fold 0): sqrt/log means over 256 or
the prefix 0.55, skewness 0.54, m5/m4 0.54, FFT bands 0.51 - weak but real
signal, all of it already captured by existing detectors.
