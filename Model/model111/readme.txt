Model111: oracle distillation inside the frozen Model84 expressive recipe.

Model109 showed soft targets from a future-seeing teacher improve a raw
LightGBM student by ~+0.007 TS-AUC, but that student lacked Model84's
synthetic splices and final stack. Model111 changes exactly one thing in
Model84's expressive voter: real-row targets become
    alpha * y + (1 - alpha) * teacher      (alpha from config.json, default 0.5)
Synthetic aligned-splice rows keep hard labels. Fit rows, 40% synthetic cell
mass, 317 fold contract columns, reference/splice weights and LightGBM settings
are Model84's (objective binary -> cross_entropy to accept soft labels).

Teachers: Model109 privileged features + the 317 fold columns, trained on
folds outside {k, k+1, j} (k held, k+1 reserved by the legacy stack), predicting
fit rows in fold j. Held fold k never reaches any teacher or student.

Final score: the Model84 postprocess chain (Model26 depth-mix replacement,
ensemble margin, anti-delta, causal smoother, 50/50 with Model67) is re-applied
with the Model111 voter, and compared with the reconstructed Model84 OOF
(0.637888).

Linux-only imports of the historical runners (fcntl, resource) are stubbed so
this runs on Windows. Requires xgboost (installed into .venv).

Run: .venv\Scripts\python.exe -u models\model111\run_model111.py 0 1
No upload or submission.
