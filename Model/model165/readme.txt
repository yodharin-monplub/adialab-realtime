Model165 - EXTRA SPLICE sets (2026-09-23).

Every student sees the same 1,000 spliced series per fold (frozen Model20
plan). Pooling extra RE-CUT sets gave the last big gain (Model163), so more
splice variety was the obvious next test; the 2025 "8x pairing" arm had
failed with hard labels only.
build_splices2.py: new pairs per fold via models/model18/augmentation
(fitting donors only, never fold k or the reserved fold k+1), assembled like
the deployed splices (aligned), 50 sampled rows each, with the 500 union
columns + 27 split tests + 32 teacher inputs so they can get soft targets.
  ~1,920 new splices per fold (reuse cap), 96k rows per fold, ~13 min/fold.
  Distribution check vs the deployed splices: median shift 0.04 IQR.
holdout_splice2.py (in memory), same recipe as Model163 + these rows in the
non-real block with oracle-teacher soft targets:
  sealed holdout  student 0.639912 (vs 0.640603), final 0.636913 (vs 0.637265)
-> no gain; slightly worse. CLOSED, not used in any bundle.
Data kept: data/splice2_fold{k}.npz.
