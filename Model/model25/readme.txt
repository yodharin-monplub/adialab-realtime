Model25: fixed rolling-peak correction sensitivity
==================================================
Model23 uses a 128-step rolling-maximum correction at weight 0.10 before its
90/10 compact blend. Test exactly 0.05 and 0.20; no new grid or weight search.
All upstream models and compact blend stay fixed. No training or submission.
References: Model20 and Model23. Same real development folds, 1000 paired
whole-series bootstrap draws, four simultaneous comparisons. Both references
must be beaten by >=0.0003, >=4/5 fold wins, and positive lower95% bounds.
Exploratory reused-CV follow-up; previous broad postprocessing study failed.
Historically revealed outer holdout is not reopened; test labels not loaded.
Freeze code and source hashes before scoring. Batch and streaming transforms
share the established causal implementation. Preserve existing models.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model25/run_model25.py run
