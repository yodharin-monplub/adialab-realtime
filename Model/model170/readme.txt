Model170 - neural student trained on ALL 10,000 labelled series (2026-09-24).

The tree student has trained on every labelled series since Model154; the
neural part still sees only the 8,333 development series.  build_all_sequences.py
adds the holdout series to the sequence file:
  inputs   the 102 GRU inputs for the holdout rows, built exactly as the bundle
           builds them (models/model153/offline_gru.gru_inputs, parity 5e-7)
  targets  dev rows: Model161's cross-fitted oracle soft targets (softB);
           holdout rows: the dev-only oracle teacher
           (models/model154/runs/hold_teacherB.txt) applied to EVERY holdout
           row - still out-of-sample for those series.
           That teacher scores 0.72317 TS-AUC on all 840,797 holdout rows.
  row_fold = -1 for holdout rows, so every fold model trains on them.
Output: data/sequences_all.npz (5,036,517 rows) + data/soft_all.npy.

HONEST CHECK: each net is still judged on the development fold it never saw,
so the all-series nets can be compared directly with the dev-only nets of
Model169 (same seed, size and epochs):
  dev-only (Model169 h256s0): fold0 0.636866  fold1 0.595178  fold2 0.628359
                              fold3 0.604497  fold4 0.620336
  all-series (runs/all_all.log): running on the CPU (~1.8 h per fold; the GPU
  would take ~6 min per fold but vast.ai credit ran out).
If the all-series nets win, the final bundle's neural part gets the same
treatment as its tree part.
