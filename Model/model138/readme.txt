Model138 - STAGE 2 of the user's protocol: series transforms.

Transforms of the history-standardized values, applied before any statistic:
identity, |x|, x^2, sign(x)*sqrt|x|, sign(x)*log(1+|x|), first difference,
rank-to-Gaussian and rank-to-Laplace (empirical CDF of the history mapped to
the normal / Laplace quantile; causal, history-fitted). Fourier features are
tests of their own in stage 3 (spec_low / spec_high / spec_entropy / spec_peak).

Evaluation: the whole stage-3 test bank (models/model139) is run on every
transform; a transform is judged by the number and strength of significant
tests it yields compared with the identity, and the best (transform, test)
pairs go to stage 4.
RESULT (20:20, full grid in models/model139/readme.txt): first differences win
for the scale / distribution / change-point tests (top-20 average 0.555 vs
0.547 identity); |x| and x^2 win for the location scans (Buishand 0.565 vs
0.534 on the raw series); sqrt, log, rank-Gaussian, rank-Laplace are within
1 s.e. of identity. Decision: diff (all 13 stage-4 tests) + abs (location
scans) become the extra stage-4 arm "ED" in models/model140.
