Model153 - full-development refit of the GRU students (2026-09-21).

Why: Model152 showed the tree student is series-limited (voters refit on all
8,333 dev series instead of 3/5: sealed holdout +0.0013 vs Submission33). The
GRU ensemble (weight 0.5 in the blend) was also trained as fold voters on 3/5
of the series. Here each deployed config is retrained on ALL dev series.

Recipe (train_gru_full.py; Model124 trainer code imported unchanged)
  inputs   models/model124/data/sequences.npz (102 standardized inputs)
  targets  0.5*y + 0.5*teacher, teacher = Model152's cross-fitted 4-fold
           teachers (models/model152/runs/full_teacher.npz)
  configs  h192 L2 16 ep lr 2e-3 b48 | h96 L3 24 ep lr 1e-3 b48 |
           h128 L2 24 ep lr 1e-3 b32 (as deployed); final epoch kept
           (no held fold; OneCycle ends at a small learning rate)
  CPU ~175-270 s/epoch with 8 torch threads -> ~4 h for the three configs.
    python models/model153/train_gru_full.py h96l3 0   (also h192, h128)

Evaluation (offline_gru.py, no new holdout pass needed)
  GRU inputs rebuilt from Model152's saved holdout columns exactly like
  Bundle.gru_input; `check` = deployed 15 GRUs reproduce the saved GRU logits
  to 5.4e-7. `score TAG STUDENT CFG...` swaps the GRU part of the blend
  (same constants as deployed, nothing tuned on the holdout).
    python models/model153/offline_gru.py score G3 BAGF3 h96l3_s0 h192_s0 h128_s0

RESULTS 2026-09-22 (raw GRU TS-AUC on the sealed holdout)
  deployed h96 L3 fold voters, single: 0.6170 0.6052 0.6181 0.6059 0.6015
                                        (mean 0.6097)
  deployed h96 L3, mean of the 5 voters: 0.6198  (averaging adds +0.010)
  full-dev h96 L3, one model:            0.6055
-> the full-data GRU is no better than a single fold voter (inside the
   seed spread 0.60-0.62). The GRU is seed-variance-limited, not
   series-limited. Matching the 15-model ensemble would need 15 full-data
   GRUs (~15 CPU hours) for no expected gain. CLOSED; h192/h128 runs
   stopped. The deployed GRU ensemble stays.
