Model172 - SYNTHETIC sequences for the neural student (2026-09-25).

WHY
The tree student draws 40% of its training weight from synthetic series (glued
splices and re-cut real series).  The neural student has only ever seen real
ones, and Model170 showed the nets are starved of SERIES, not of capacity:
adding the 1,667 holdout series to their training set gained +0.0069 dev, 5/5,
while more hidden units (384) and more epochs gained nothing (see below).
The synthetic pool is far bigger than that: 7,788 re-cuts + 5,000 splices.
So: give the nets the same synthetic series the tree student already uses.

FILES
build_synth_sequences.py  full per-step sequences for the synthetic series in
    the GRU's own input space (102 contract inputs, standardized and clipped
    exactly like models/model124/export_data.py), targets 0.5*y + 0.5*teacher
    with the Model152 oracle teacher of a fold the parent series is NOT in.
      python models/model172/build_synth_sequences.py recuts  --workers 5
      python models/model172/build_synth_sequences.py splices --workers 5
    -> data/seq_recuts.npz (2,538,463 rows), data/seq_splices.npz (2,874,804)
train_gpu2.py             Model169's training loop (= Model124's deployed
    recipe) reading SEVERAL sequence files: the first is the real one
    (models/model170/data/sequences_all.npz), the rest synthetic, with
    --synth-share 0.4 giving the synthetic rows 40% of the sample weight.
    A synthetic series is dropped from fold k's training set when its parent's
    row_fold is k or the stack-reserved fold (k+1)%5 - the tree student's rule.
    Epoch selection stays on the held development fold, real rows only, so the
    numbers compare directly with Model169/170.
      python train_gpu2.py --fold 0 --hidden 256 --epochs 24 \
        --data sequences_all.npz,seq_recuts.npz --synth-share 0.4 --tag synR_

EPOCH PROBE (ran 2026-09-25 03:44-04:22 on the rented RTX 3090, all-series
real rows only, so this is Model170's arm with a longer schedule)
  held development fold          E24 (Model170)   E36      E48
    fold 0                         0.637716     0.638846  0.638169
    fold 1                         0.610761     0.610366  0.612460
    fold 2                         0.631663     0.636334  0.636181
    fold 3                         0.609817     0.612043  0.609067
    fold 4                         0.629898     0.627409  0.625643
    mean                           0.623971     0.625000  0.624304
  E36 is +0.0010 over E24 but only 3/5 folds and two folds are negative:
  noise-level by the project rule (gain > 2 s.e. AND >= 4/5 folds).  Longer
  training is NOT the lever; keep 24 epochs.  Logs: runs/train_e36.log,
  runs/train_e48.log.

DATA LOSS AND RECOVERY (2026-09-25)
The project folder was moved to ...\coding\structural-break-real-time-minimum-mantis
while an earlier session was running.  Three caches did not survive the move:
models/model170/data/sequences_all.npz, soft_all.npy and this model's
data/seq_recuts.npz.  sequences_all.npz and soft_all.npy were still on the
rented GPU box and were downloaded back; seq_recuts.npz had been truncated
mid-upload (BadZipFile on the box) and is being rebuilt locally by
build_synth_sequences.py (~42 min, 5 workers).  build.log is the first build,
build_recuts2.log the rebuild.

STATUS
Data rebuild running; the synR_ arm (real + re-cuts) trains next on the box
(~6 min per fold), then the splice arm.  Decision rule: the arm must beat
Model170's nets by > 2 s.e. on >= 4/5 dev folds before it goes anywhere near a
bundle; then the Model171 weight protocol picks ws/wg, and only then a
Submission43 is built for the user's approval.
