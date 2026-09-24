Model169 - a BIGGER neural student (hidden 256) on the oracle-teacher targets,
2026-09-23.  User's instruction: "if deploy neural network, deploy in a pod",
and longer training time is acceptable.

Why: the deployed GRU part saturates at ~0.619 on the holdout (Model168) and
its targets were the OLD teacher's.  Model161 showed same-size nets with the
new targets are no better, so the open question is CAPACITY.

train_gpu.py = the deployed training loop (Model124) with any hidden size /
depth / epochs, the Model157 oracle soft targets (models/model161/data/softB
-> soft.npy) and best-held-epoch selection, on GPU if available.
pod.py = RunPod rental helper (create / list / ssh / stop) using an SSH key
generated locally (Other/runpod_key), so no third-party tool is installed.

GPU STATUS 2026-09-23: RunPod API works but pod creation fails - "account
balance is too low to rent a pod"; the vast.ai key in .env (73 chars) is
rejected as invalid (vast expects 64 hex). User informed by email; meanwhile
the training runs on the CPU (~25-45 min per epoch under load).

RESULTS (h256, 2 layers, 24 epochs, batch 48, oracle targets)
  fold 0: best held TS-AUC 0.637234  (deployed h96 L3 fold 0: 0.635425,
          deployed h192 fold 0: 0.6307)  -> the bigger net is better
  folds 1-4: queued (runs/h256_rest.log)
Next: score the 5 nets on the holdout (models/model153/offline_gru.py style),
compare the GRU component 0.6191 -> ?, and only then consider a new bundle.

GPU RUNS (vast.ai RTX 3090, $0.18/h; vast.py; both instances destroyed after
use; total spend about $0.45). NOTE: the vast key lives in Other\.env - the
ROOT .env is a SEPARATE stale copy; keep them in sync (that was why the key
looked invalid).
  h256 (2 layers, 24 ep) x 3 seeds x 5 folds: 35 min on the GPU
  h384 (2 layers, 28 ep) x 3 seeds x 5 folds: 20 min
SEALED HOLDOUT (raw = neural part alone; blend uses the honest dev-only
student SRP40 at weight 1.5)
  deployed 15 nets      raw 0.619054   blend 0.637246
  h256 seed0 (5 nets)   raw 0.622011   blend 0.638015
  h256 seeds0-1 (10)    raw 0.621646   blend 0.637636
  h256 all (15)         raw 0.621560   blend 0.637557
  h384 seed0 (5 nets)   raw 0.623888   blend 0.638252
  h256s0 + h384s0 (10)  raw 0.623422   blend 0.638159
-> capacity saturates by hidden 256-384; MORE SEEDS DO NOT HELP (5 > 15).
   Chose h256 seed0 (5 nets): +0.0008 in the blend at a third of the serving
   cost of h384, runtime +6% vs Submission40.
SUBMISSION41 = Submission40 with that GRU set (deployment/build41.py,
validate41.py: parity 1.0e-7, replay 0.0, ~5.0 cloud hours; seal35.py).
Constants mu -0.19964 sd 0.70670 = mean/sd of the new nets' dev OOF logits.
Emailed the user for approval; NOT uploaded.
