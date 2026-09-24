Model 16: persistence plus compact diversity

Fixed margin = 0.9 * Model 10's rolling-max-corrected Model 9 margin
               + 0.1 * Model 11 margin.
The rolling correction uses window 128 and correction weight 0.1.
Run: OPENBLAS_NUM_THREADS=1 python3 models/model16/run_model16.py

This combines two reused-development diagnostics and is explicitly post-hoc.
Model 10 rejected promotion of its transform. Model 9's base refitting is not
fully nested. A higher local score does not establish external improvement.
No outer-holdout or test labels are loaded; no model is automatically deployed.
