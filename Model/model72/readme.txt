Model72: fold-local LightGBM gain pruning from 317 to 160 causal features.

Models64, 66, 69 and 71 expanded the feature set and failed to improve the
strong Model62 voter. A read-only audit of Model62's frozen fold boosters
showed that the top 160 of 317 features account for about 95% of split gain
in every fold. Model72 fixes 160 before scoring. For each held fold, it
selects the 160 largest-gain features from that fold's Model62 booster,
which was fitted without that held fold or the adjacent stopping fold.
Ties are broken by original feature index; selected columns retain their
original order for exact real/synthetic/inference alignment. This is
fold-local training-only selection, not held-score selection.

The same 400,000 real and 400,000 aligned-positive synthetic fitting rows,
40% synthetic mass, 800-tree LightGBM recipe, seeds, five series-disjoint
folds and frozen Model39 stack/postprocess are retained. No raw features
are recomputed; verified Model20/57 caches provide both feature matrices.

Four voter blends are prespecified: new-voter fractions 0.50/1.00 and
replacement weights 0.50/1.00. Pilot folds 0-1 need mean gain >=+0.0001
versus Model62 before confirmation folds 2-4 train. Promotion requires
confirmation and full held-series OOF TSAUC >=0.640, confirmation gain
>=+0.0005, at least 2/3 wins, positive paired-series bootstrap lower 95%
bound, and positive gains in both fixed ID halves. Scores are exploratory
because inherited features and ensemble choices used development data.
The sealed outer holdout and competition test labels remain closed.

CPU only, four LightGBM threads, 15-hour local timeout, fold checkpoints.
No automatic packaging, upload or cloud submission.

Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model72-mpl .venv/bin/python -u models/model72/run_model72.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model72-mpl .venv/bin/python -u models/model72/run_model72.py report
