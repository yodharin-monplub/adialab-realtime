Model127: GRU student on the 61 whitening channels only (maximum diversity).

Model126 showed the GRU's blend value comes from being different from the
distilled tree, not from raw strength (wider tree-like inputs hurt the blend).
Model127 goes the other way: inputs = Model57's 61 history-whitened columns +
log1p(t) only (62 inputs), same fold-clean distilled targets, h96 3-layer GRU,
12 epochs, local CPU. Pilot folds 0-1; compared alone and as a 4th member of
the Submission31 ensemble with models/model124/blend_eval.py.

Run:
  .venv\Scripts\python.exe models\model124\export_data.py 0 sequences_d0
  (cwd models\model127\runs, PYTHONPATH=project root)
  python ..\..\model124\train_gru.py --data ..\..\model124\data\sequences_d0.npz --fold K --hidden 96 --layers 3 --epochs 12 --lr 1e-3 --tag d0_
Pilot (raw held-fold TS-AUC, 12 epochs, ~140 s/epoch on CPU)
fold 0  0.596466      fold 1  0.577448      (much weaker raw than the 102-input GRU)
Top-level blend on Submission30 (blend_eval.py):
  alone, fold 0: +0.0012 at w=0.35 - as much as the far stronger 102-input GRU;
  as 4th member of the Submission31 ensemble (h192, h96l3, h128 + d0):
  pilot +0.00216 at w=0.5 vs +0.00180 for the 3-config ensemble, both folds up.
Conclusion so far: the GRU's value is diversity from the tree; the whitening-
only model is the most diverse and helps most per unit of raw strength.
Confirmation folds (raw): fold 2 0.596153, fold 3 0.566181, fold 4 0.589941
(five-fold raw mean 0.5852 - the weakest model in the ensemble by far).
Submission32 = Submission30 + 4-config GRU ensemble (h192, h96l3, h128 + this
whitening-only GRU; 20 numpy GRUs, two input contracts; select_gru32.py):
  w_gru = 0.5 (pilot folds 0-1); pilot +0.00224, confirmation +0.00214 (3/3)
  vs Submission30; dev OOF 0.639083 vs Submission31 0.638852 (+0.00023).
  Parity (validate31.py, SUBMISSION_NAME=submission32): 1.8e-7, deterministic,
  incremental, train() data-free. Benchmark 1.10x Sub30 (~4.3 h). Sealed.
SEALED HOLDOUT: Submission32 0.632011 vs Submission31 0.632221 (-0.0002) vs
Submission30 0.630638. The +0.0002 dev improvement did not transfer; the
4th (whitening-only) member adds nothing on untouched data. Submission32 is
NOT submitted; Submission31 remains the candidate. Closed.
