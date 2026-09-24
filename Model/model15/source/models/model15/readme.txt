Model 15: residual boosting

Learn a conservative LightGBM correction to Model 9's frozen OOF margin,
using Model 11's compact causal features plus the current Model 9 margin.
CPU only, fixed 150 rounds, seven leaves, large leaves and L2 regularization.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python3 models/model15/run_model15.py
All corrections are evaluated on complete-series held folds. No outer holdout
or test labels are loaded. Model 9's upstream OOF limitations remain, so this
is exploratory stacking, not confirmation or an automatically deployed model.
