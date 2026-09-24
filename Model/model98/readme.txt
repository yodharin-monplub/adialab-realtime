Model98: residual Bayesian scale evidence on the locked scale-GLR correction.

This is the remaining compact combination implied by the Discord recipe. The
Model95 maximum-over-tau Gaussian scale GLR and Model90 uniformly tau-integrated
inverse-gamma Bayes evidence summarize different alternatives: the former is
strong when one candidate break dominates, while the latter accumulates weaker
evidence across candidate break times and amplitudes.

The Model95 GLR coefficient is frozen at its pilot-selected value 0.02. On
pilot folds 0-1 only, Model98 tests additive Bayesian coefficients
-0.04, -0.02, 0, 0.01, 0.02, and 0.04 on top of the GLR-corrected Model84
native score. Including zero makes the locked Model96 signal the baseline;
negative coefficients test whether diffuse Bayesian evidence mainly removes
false-positive GLR peaks. Confirmation folds stay untouched by this decision.

Continue only for a pilot gain of at least 0.0001 over locked Model96. If it
passes, a separate Model99 will freeze the chosen coefficient and report the
three confirmation folds, series bootstrap, and fixed ID halves. CPU only;
no cloud packaging, upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model98/run_model98.py
