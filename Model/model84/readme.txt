Model84: LightGBM capacity-diversity test on the proven 40%-synthetic pool.

Model62's standard 31-leaf LightGBM is strong, while Model63 shows that learner
diversity can improve the final ensemble. Model84 reuses the exact frozen
400,000 real rows, 400,000 aligned-splice rows, 40% cell-wise synthetic mass,
317 causal features, exclusions, and postprocessing. It changes only the tree
recipe. The smooth arm uses 15 leaves, larger leaves, and slower boosting; the
expressive arm uses 63 leaves and smaller leaves. This brackets the incumbent
capacity instead of adding another feature bank or synthetic cache.

Both arms train on pilot folds 0-1. Fixed blend weights 0.125, 0.25, and 0.50
mix each processed voter into Model67. The best pilot choice must gain at least
0.0001 before its arm is trained on folds 2-4. Promotion requires confirmation
and full OOF TSAUC >=0.640, confirmation gain >=0.0005, 2/3 wins, positive
paired-series bootstrap lower 95%, and positive fixed-ID-half gains.

CPU-only, four LightGBM threads, checkpointed per fold and arm. No cloud run,
packaging, upload, or submission.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model84-mpl .venv/bin/python -u models/model84/run_model84.py
