Model 16: persistence plus compact diversity

Fixed margin = 0.9 * Model 10's rolling-max-corrected Model 9 margin
               + 0.1 * Model 11 margin.
The rolling correction uses window 128 and correction weight 0.1.
Run: OPENBLAS_NUM_THREADS=1 python3 models/model16/run_model16.py

This combines two reused-development diagnostics and is explicitly post-hoc.
Model 10 rejected promotion of its transform. Model 9's base refitting is not
fully nested. A higher local score does not establish external improvement.
No outer-holdout or test labels are loaded; no model is automatically deployed.

Submission preparation (2026-09-06):
  Bundle: submissions/submission16/; archive: submissions/submission16.zip
  Entry: main.py; saved model directory: resources/
  INFER_PARALLELISM = 12; one numerical-library thread per worker.
  Cloud: gcp-cpu, 16 vCPU / 64 GiB; forceFirstTrain=false; trainFrequency=0.
  Expected cloud execution: about 4 hours (planning range 3-5 hours, no queue).
  See SUBMISSION16_ESTIMATE.md, cloud_run_settings.json, and submission_validation/.
  Export parity, deterministic streaming, and Python 3.12 inference passed.
  No upload or cloud run was performed as part of preparation.

Subsequent user-requested push:
  Accepted as Crunch platform submission 10, submission ID 78615.
  69 source files and 16 pretrained model files uploaded.
  Receipt: submission_validation/push_receipt.json
  cloud_run_settings.json now includes the uploaded submission ID.
  The push did not start a cloud run.
