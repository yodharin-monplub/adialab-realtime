Model136: monotone-in-time score constraint (user suggestion, 2026-09-18 02:53).

The label y_t is non-decreasing within a series, so the user asked whether
forcing the score to be non-decreasing helps. The deployed chain already has
a windowed version (rolling max over 128 steps, 10% weight, Model16/26 era,
about +0.001). monotone_probe.py applies the full constraint to Submission31's
scores (dev OOF, five folds; and the sealed-holdout predictions):

  transform                          dev mean    holdout
  cumulative max, 100% (monotone)    -0.0378     -0.0424
  75% towards cummax                 -0.0229        -
  50%                                -0.0101     -0.0118
  25%                                -0.0019     -0.0021
  10%                                -0.0001     -0.0001
  ratchet envelope decay 0.9/.97/.99 -0.0003 / -0.0020 / -0.0060 (holdout -0.0007/-0.0036/-0.0097)

Conclusion: the optimal score is NOT monotone. P(break already happened |
data so far) legitimately falls when a suspected break is revised away, and
TS-AUC ranks series within each time step, so keeping a stale maximum pushes
false alarms above true breaks. Closed 03:00.
