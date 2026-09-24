Model148 - the GRU component with the 27 whitened split-test features as extra
inputs (2026-09-21). Motivation: those features gave the tree student +0.0037
on the dev folds and, combined with re-cut data and bagging, +0.0023 on the
sealed holdout (Model147 BAG3); the GRU component of Submission31 has never
seen them.
export_split_inputs.py: Model124's export (102 standardized inputs, soft
targets, folds) + the 27 split columns standardized / clipped the same way ->
data/sequences_split.npz (129 inputs). Training uses the unchanged
models/model124/train_gru.py (same targets, weights, folds, best-epoch rule).
Evaluation: models/model124/blend_eval.py-style check of the GRU's value on
top of the blend, against the existing GRU of the same size trained on 102
inputs (same epochs, same CPU) - one factor changed.
RESULTS 2026-09-21 (h96l3s_ = hidden 96, 3 layers, 24 epochs, lr 1e-3, batch 48,
CPU ~135 s/epoch; train with PYTHONPATH=<root> so train_gru.py finds tsauc.py)
  raw held TS-AUC, old h96l3 (102 inputs) -> new (129 inputs):
    0.6354->0.6424  0.6020->0.6113  0.6281->0.6281  0.6006->0.6044  0.6170->0.6205
    mean 0.6166 -> 0.6213 (+0.0047)
  inside the full blend z(chain) + .5 z(BAG3 student) + .5 z(GRU set)
  (blend_eval148.py): swapping this ONE of the three GRU configs
    +0.0008 +0.0010 +0.0001 +0.0001 +0.0006   mean +0.0005, 5/5
    (adding it as a 4th config: +0.0004, 4/5)
  dev OOF of the whole blend with the swap: 0.64107 (Submission31 was 0.63885).
weights_eval.py: the top-level weights (student .5, GRU .5) are still right -
no pair selected on the pilot folds survives confirmation (best: ws 1.0 /
wg .75, pilot +0.0005, confirmation -0.0003); without the chain -0.0042.
ALL THREE SIZES (2026-09-22; select_gru148.py -> constants_gru148.json):
  raw held TS-AUC per fold
    h96l3s_ 0.6424 0.6113 0.6281 0.6044 0.6205
    h128s_  0.6429 0.6109 0.6295 0.6076 0.6304
    h192s_  0.6370 0.6043 0.6294 0.6128 0.6220
  ensemble (logit mean): old set 0.6208 -> new set 0.6281 (+0.0073)
  full blend z(chain) + .5 z(BAG3 student) + .5 z(GRU set):
    old 0.64059 -> new 0.64190; gain +0.0027 +0.0026 +0.0003 +0.0003 +0.0006,
    mean +0.0013, s.e. 0.0006, 5/5.
deployment/: solution148.py (extends solution147: the 27 split columns are
computed once per step and shared by the student and the GRUs), build34.py
(Submission34 = sealed Submission33 + 15 new GRUs + contract + training code),
parity_inputs.py: the 129 inputs built step by step by the bundle equal the
training export rows exactly (0.0 on 4 series / 2,595 steps).
holdout_eval_bundle.py runs the BUILT bundle's own SeriesPredictor on the
1,667 sealed series (resumable); Submission34 run started 2026-09-22,
holdout_submission34.log (~3 h).

SEALED HOLDOUT OF THE BUILT BUNDLE (2026-09-22, 8,084 s):
  Submission31 0.63222 | Submission33 0.63448 | Submission34 0.63354
  -> +0.0013 vs Sub31 but -0.0009 vs Sub33: the GRU upgrade does NOT transfer.
Likely reasons: (1) each GRU keeps its best epoch as judged on the held fold,
and with the richer inputs the networks overfit faster (training loss keeps
falling while the held score peaks mid-training), so the dev numbers are more
optimistic than for the old GRUs; (2) GRUs that see the split tests agree more
with the student, which already uses them, so the blend loses diversity.
DECISION: Submission33 stays the upload candidate; Submission34 is kept on disk
unsealed, not uploaded. Closed.

Earlier plan (started 2026-09-21 evening, ~9 h CPU): the other two sizes with the same
inputs - h128s_ (128, 2 layers, 24 epochs, lr 1e-3, batch 32) and h192s_ (192,
2 layers, 16 epochs, lr 2e-3, batch 48); then the 3-config set vs the old set.
