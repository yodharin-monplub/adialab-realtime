Model49: larger real-data fit for the proven synthetic XGBoost voter.

Model20's backbone caps real fitting rows at 400,000 per fold, roughly 16% of
the approximately 2.5 million eligible rows. Model49 keeps those exact
400,000 rows and deterministically adds 800,000 distinct real fitting rows
from the same three permitted folds. The 400,000 aligned-positive synthetic
examples, 256 fold-specific causal detector features, XGBoost depth-3 recipe,
800 rounds, and mass-preserving 25% synthetic weight share remain fixed.
Training and donor sources exclude the held fold and adjacent stopping fold.

One new voter is fitted on folds 0-1 first. Pilot weights 0.05, 0.10, 0.20,
0.40, 0.80 replace the Model26 XGBoost column before the frozen ensemble,
Model35 anti-residual correction, and Model38 causal smoother. If the best
pilot mean gain over Model39 is below +0.0001, stop. Otherwise keep that
weight fixed for unseen confirmation folds 2-4. Success requires both the
confirmation and full five-fold mean TSAUC >=0.640, confirmation gain
>=0.0005, at least 2/3 confirmation wins, positive 95% whole-series bootstrap
lower bound, and positive gains in two fixed disjoint series-ID halves.

This is exploratory held-series OOF because upstream feature selection and
ensemble weights were development-informed. Outer holdout and test labels
remain closed. No packaging, upload or cloud run is automatic.

CPU-only: four XGBoost threads. Expected about 8-12 minutes per fit; memory
use can be several GiB. Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model49/run_model49.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u models/model49/run_model49.py report
