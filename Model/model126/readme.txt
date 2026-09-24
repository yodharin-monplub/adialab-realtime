Model126: GRU student with 200 detector columns (Model124 with a wider input).

Model124's GRU sees 61 whitening + 40 detector columns; the distilled tree
uses 256 detector + 61 whitening columns and is 0.003 stronger raw. Model126
re-exports the sequences with the top-200 detector columns (Model109 fold-0
hard-student gain order) -> 262 inputs, same fold-clean distilled targets, and
trains the h96 3-layer GRU on the local CPU (the GPU instance was destroyed;
~2-3 min per epoch locally). Pilot folds 0-1 first, 12 epochs.

Run:
  .venv\Scripts\python.exe models\model124\export_data.py 200 sequences_d200
  (cwd models\model126\runs, PYTHONPATH=project root)
  python ..\..\model124\train_gru.py --data ..\..\model124\data\sequences_d200.npz --fold K --hidden 96 --layers 3 --epochs 12 --lr 1e-3 --tag d200_
Compare raw with Model124 h96l3 (0.6354 / 0.6020 on folds 0 / 1) and the
blend on Submission30 with models/model124/blend_eval.py.
Pilot result (raw held-fold TS-AUC, 12 epochs, 144 s/epoch on CPU)
fold      Model126 (262 inputs)   Model124 h96l3 (102 inputs, 24 epochs)
0         0.633698                0.635425
1         0.598502                0.602041
Top-level blend on Submission30 (blend_eval.py): alone, pilot +0.0006 at
w=0.25 (the 40-column GRU gives +0.0018); as a 4th ensemble member, pilot
+0.00169 vs +0.00180 without it. Wider, more tree-like inputs make the GRU
more correlated with the distilled tree and LESS useful in the blend: the
GRU's value is diversity, not raw strength. Closed; not deployed.
