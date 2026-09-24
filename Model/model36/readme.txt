Model36 — multi-order AR mean-shift boundary likelihood features

The July 2026 AR(p)-FOCuS paper derives a likelihood-ratio detector for mean
changes in autocorrelated streams. The incumbent already contains AR(5)
residual CUSUMs, so this experiment isolates the genuinely different part:
the exact boundary-response weights induced when the process mean changes.

For AR orders 1, 2, 5 and 10 fitted only on clean history, six causal suffix
ages (8–256) produce signed and squared normalized likelihood scores plus
per-order maxima. These 60 columns are appended to the same fold-specific 256
features used by Model20. One depth-3, 800-round CPU XGBoost is fit per held
series fold on the identical 400,000 real rows and reference weights.

The primary candidate mixes 20% of this enhanced component with 80% of the
Model26 component before applying the frozen stack/post-process. Promotion
requires held-series OOF TSAUC >= .640, gain >= .0007, 4/5 wins, a positive
whole-series bootstrap lower bound and positive fixed ID-half gains. No test
labels, outer holdout, cloud training, or automatic upload are used.
