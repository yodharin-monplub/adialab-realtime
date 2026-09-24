Model63: interaction test, extra-randomized LightGBM at 40% synthetic mass.

Model61 changed the LightGBM split rule to extra_trees at 25% synthetic mass
and improved the difficult fold 3, while Model62 changed synthetic mass to
40% under standard splits and improved fold 1. Their two changes are
combined in one new learner. The same 400,000 real and 400,000 aligned
synthetic rows, 256 selected original plus 61 whitened causal features,
donor exclusions, 800-tree/31-leaf recipe, seeds and four CPU threads are
retained. Model62's verified feature caches and training implementation
are reused read-only; the only recipe difference is extra_trees=True with
extra_seed=61, exactly as in Model61.

The candidate is either the new voter's probabilities alone or their 50/50
average with Model62's frozen standard-split 40%-mass voter. Each replaces
0.50 or 1.00 of the Model26 voter before the unchanged Model39 stack and
causal smoother. Four choices are frozen before fitting. Pilot folds 0-1
select one; gain must be >=0.0001 to train confirmation folds 2-4.
Promotion requires confirmation and full five-fold TSAUC >=0.640,
confirmation gain >=0.0005, 2/3 wins, positive paired-series bootstrap
lower 95% bound, and positive gains in both fixed ID halves.

These are exploratory development held-series OOF scores. The sealed outer
holdout and competition test labels remain closed. No cloud packaging,
upload or submission is automatic. Expected local runtime is well below
the user's 10-hour limit; the process is capped at 9 hours.

Run/resume from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model63-mpl timeout 9h .venv/bin/python -u models/model63/run_model63.py run
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model63-mpl .venv/bin/python -u models/model63/run_model63.py report
