Model128: GRU student on raw standardized series values (no engineered features).

Models 126/127 showed the GRU's blend value is diversity from the distilled
tree, and that the least tree-like input (whitening only) helped most per
unit of raw strength, yet Submission32 did not transfer to the holdout.
Model128 tests the most different signal source available: the raw online
values themselves (v, v^2, |v|, sign v, previous v, log1p t -> 6 inputs), so
the GRU must learn its own change detectors. Same fold-clean distilled
targets, h96 3-layer GRU, 12 epochs, local CPU. Pilot folds 0-1; evaluated
alone and as an extra member of the Submission31 ensemble with
models/model124/blend_eval.py. Promotion would still require a sealed-holdout
gain over Submission31 (the arbiter that just rejected Submission32).

Run:
  .venv\Scripts\python.exe models\model128\export_raw.py
  (cwd models\model128\runs, PYTHONPATH=project root)
  python ..\..\model124\train_gru.py --data ..\..\model124\data\sequences_raw.npz --fold K --hidden 96 --layers 3 --epochs 12 --lr 1e-3 --tag raw_
Result: fold 0 held TS-AUC 0.508 (epoch 2), 0.513 (epoch 6) - essentially
chance. A from-scratch GRU on raw values does not learn change detection at
this size/epoch budget (the engineered whitening/detector features carry the
signal). Stopped at epoch 6 (14:12); fold 1 not run. Closed.
