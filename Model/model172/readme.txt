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

RESULT - ARM synR (real all-series + 7,788 re-cut sequences, 40% weight)
2026-09-25, RTX 3090, 24 epochs, hidden 256, one seed x five folds.
Compared with Model170's all-series nets, the SAME recipe without the
synthetic series (this is the only thing that changed):
  held fold      Model170    synR       gain
    0            0.637716   0.643048   +0.0053
    1            0.610761   0.599770   -0.0110
    2            0.631663   0.626522   -0.0051
    3            0.609817   0.609061   -0.0008
    4            0.629898   0.632237   +0.0023
    mean         0.623971   0.622128   -0.0018 +/- 0.0029, wins 2/5
-> FAILS the project rule (needs > 2 s.e. and >= 4/5 folds). The effect is not
a small loss but a large VARIANCE: +0.0053 on fold 0, -0.0110 on fold 1.
Reading: the tree student sees each step independently, so a re-cut series is
just more rows; the net carries hidden state across the whole sequence, and a
re-cut series is a real series whose history was truncated and whose break was
moved, which distorts the time/state prior the net builds up. Model170's gain
came from more REAL series, not from more rows.
Follow-ups running: arm synR15 (the same data at a 15% weight share, to test
dosage) and the splice sequences (splices keep the deployed alignment, so they
may disturb the sequence prior less than re-cuts do).
Logs: runs/train_synR.log, predictions runs/gpu/synR_fold{k}_pred.npz.

RESULT - ARM synR15 (the same re-cut sequences at a 15% weight share)
  held fold      Model170    synR15     gain
    0            0.637716   0.634923   -0.0028
    1            0.610761   0.605462   -0.0053
    2            0.631663   0.623536   -0.0081
    3            0.609817   0.609365   -0.0005
    4            0.629898   0.631150   +0.0013
    mean         0.623971   0.620887   -0.0031 +/- 0.0017, wins 1/5
-> also FAILS, and no better than 40%: the dose is not the problem, synthetic
series are simply the wrong food for a recurrent net.
BLEND CHECK (models/model172/compare_blends.py, absolute dev blend means on one
scale, pilot-selected weights): Model170 nets 0.64325 (ws 1.5, wg 1.0) vs synR
nets 0.64269 (ws 1.5, wg 0.75) = -0.0006. So the loss is not rescued by
diversity either. Submission42's nets stay.
Still queued on the box: synRS (re-cuts + splices) and synS (splices only) -
splices keep the deployed time alignment, which is the one version of the idea
the alignment reading does not already rule out.

LEAK FOUND AND FIXED 2026-09-25 (the important entry in this file)
The first splice arm (synRS: real + re-cuts + splices) scored 0.655143 on held
fold 0 - **+0.0174** over the baseline - and 0.616742 on fold 1 (+0.006).
Gains of that size do not happen in this project, so the run was killed before
it finished and the cause was found:
  models/model18/augmentation.select_pairs drops a donor only when the donor's
  series is in the plan's OWN held fold. So plan v's spliced series may contain
  real post-break tails of every other fold, including fold k.
  train_gpu2.py keeps a synthetic series when row_fold is neither the held fold
  k nor the reserved fold (k+1)%5. With splices tagged by PLAN index that keeps
  plans v != k - precisely the plans that can carry fold-k donors. The net was
  being shown real post-break data from the series it was scored on.
The deployed tree student never had this problem: for fold k it uses plan k only.
FIX (models/model172/split_splices.py, no feature recompute needed): split the
splice sequences into one file per plan, seq_splices_p{v}.npz, with row_fold set
to -1 ("always keep"), and train fold k with ONLY seq_splices_p{k}.npz. Then the
splices a fold trains on are exactly the ones whose donors exclude that fold.
The leaky log is kept as runs/train_synRS_LEAKY.log for the record; those two
fold numbers must never be quoted as results.
NOTE the re-cut arms (synR, synR15) are NOT affected: a re-cut series carries its
parent's fold in row_fold, so the standard rule already excludes held-fold
parents. Their negative results stand.

RESULT - ARM synS (real all-series + the 5,000 SPLICED series, fold-safe:
fold k trained on plan k only, 40% weight share)
  held fold      Model170    synS       gain
    0            0.637716   0.635165   -0.0026
    1            0.610761   0.609506   -0.0013
    2            0.631663   0.628272   -0.0034
    3            0.609817   0.605092   -0.0047
    4            0.629898   0.617084   -0.0128
    mean         0.623971   0.619024   -0.0049 +/- 0.0020, wins 0/5
-> FAILS, 0/5. For the record, the LEAKY version of this same arm had scored
0.6551 on fold 0 where the honest number is 0.6352: the leak was worth about
+0.020, which is why it was caught.
The alignment reading survives only in part: splices (which keep the deployed
alignment) do less damage per fold than re-cuts on folds 0-3, but fold 4 loses
-0.0128 and the mean is worse. Conclusion for the neural student: SYNTHETIC
SERIES OF ANY KIND HURT IT. Model170's lesson stands - it wanted more REAL
series, and there are no more labelled real series in this competition's data.

RESULT - ARM synRS2 (re-cuts AND fold-safe splices together, 40% share)
  per-fold gain vs Model170: [0.0046, -0.0074, 0.0018, 0.0027, 0.0032]
  mean +0.00098 +/- 0.00215, wins 4/5
-> passes the 4/5 half of the rule and FAILS the size half (needs > 2 s.e. =
+0.0043). Do not be tempted by the 4/5: re-cuts alone were -0.0018 (2/5) and
splices alone -0.0049 (0/5), so "both together help" has no mechanism behind it
and is fold noise. With four arms x five folds, one arm looking 4/5 positive by
chance is expected; that is exactly what the 2-s.e. half of the rule is for.
