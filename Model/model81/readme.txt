Model81: whitened, metric-weighted pairwise ranking voter.

Model48 tested XGBoost rank:pairwise on 256 detector features and 100,000
synthetic rows. It improved pilot folds but did not confirm. Model81 upgrades
that distinct learner without changing the incumbent: it appends Model57's
61 history-whitened causal features (317 total), uses 266,667 aligned-splice
rows beside 400,000 real rows so synthetic examples form 40% of rows, and
weights each online-time query from n_positive*n_negative measured only on
the complete fitting-fold real population. Four mean-sampled pairs per row
and 600 depth-3 rounds are frozen.

The ranker's arbitrary margin and Model67 are each converted to percentile
ranks within online time before combining, exactly matching TSAUC's
cross-sectional comparison geometry. Ranker fractions 0.05, 0.10, 0.20 and
0.40 are frozen. Pilot folds 0-1 select one and require +0.0001 mean TSAUC.
Only a pass trains folds 2-4. Promotion also requires full and confirmation
TSAUC >=0.640, confirmation gain >=0.0005, 2/3 wins, a positive paired-series
bootstrap lower 95% bound, and positive gains in both fixed ID halves.

All features are causal; fitting rows and synthetic donors exclude held and
adjacent stopping folds. Inherited feature selection, parents and the ranker
direction are development-informed, so results remain exploratory held-series
OOF. The sealed outer holdout and competition labels stay closed. CPU only;
no cloud packaging, upload or submission is automatic.

Implementation basis:
  https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
  https://github.com/crunchdao/competitions/blob/master/competitions/structural-break-real-time/scoring/scoring.py

Run/resume from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model81-mpl timeout 9h .venv/bin/python -u models/model81/run_model81.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model81-mpl .venv/bin/python -u models/model81/run_model81.py report

