Model173 - MIRRORED real series for the neural student (2026-09-25).

WHY
Model170: the nets gain +0.0069 (5/5) from more REAL series.
Model172: they LOSE from synthetic re-cut series (-0.0018, 2/5; and -0.0031 at
a smaller dose). A re-cut series has its history truncated and its break moved,
so it breaks the time/state prior a recurrent net builds along the sequence,
while the tree student - which sees each step on its own - is happy with it.
Mirroring breaks neither: for a real series with history mean m,
    x -> 2*m - x
keeps the history length, the break time and the label, and only reverses
direction, so every signed feature (up/down CUSUM, signed t, tail_up/tail_down)
sees the opposite sign. Model150 found mirrored re-cuts made the best single
TREE student measured in this project (+0.0088 dev, 5/5), so the transform does
carry information; here it is applied to whole real series instead.

WHAT
build_flip_sequences.py mirrors each of the 8,333 development series (the real pool
of models/model20/features holds only those; the sealed-holdout series would need
Model154's holdout pool and can be added later) and
builds full per-step sequences with Model172's machinery (imported, not copied:
the 102 contract inputs, standardized and clipped as models/model124 did, and
0.5*y + 0.5*oracle-teacher soft targets, teacher j never having seen fold j).
row_fold of a mirror is its parent's fold, so fold k's net never sees the mirror
of a series it is scored on, nor the stack-reserved fold's mirrors.
  python models/model173/build_flip_sequences.py --workers 5
  -> data/seq_flip.npz  (same keys as models/model172/data/seq_*.npz)
Then, on the rented GPU (models/model172/train_gpu2.py):
  --data sequences_all.npz,seq_flip.npz --synth-share 0.4 --tag synF_

DECISION RULE
The arm must beat Model170's nets (per-fold 0.637716 0.610761 0.631663 0.609817
0.629898, mean 0.623971) by more than 2 s.e. on at least 4 of 5 held folds, and
then win on the absolute blend scale of models/model172/compare_blends.py,
before any Submission43 is built. Nothing is uploaded without the user's word.

STATUS
Builder written; runs as soon as the splice build frees the CPU.

RESULT 2026-09-25 (RTX 3090, 24 epochs, hidden 256, one seed x five folds,
mirrors at a 40% weight share; baseline = Model170's nets, the ones in Sub42)
  held fold      Model170    synF       gain
    0            0.637716   0.637246   -0.0005
    1            0.610761   0.603805   -0.0070
    2            0.631663   0.635458   +0.0038
    3            0.609817   0.608182   -0.0016
    4            0.629898   0.635916   +0.0060
    mean         0.623971   0.624121   +0.00015 +/- 0.00226, wins 2/5
-> FAILS (needs > 2 s.e. and >= 4/5). Mirroring is the best-motivated
augmentation available - it keeps history length, break time and label, and only
reverses direction - and it is FLAT. Together with Model172's arms (re-cuts
-0.0018, splices -0.0049, both +0.0010 at 4/5 but far inside noise) the
conclusion is firm:

  THE NEURAL STUDENT CANNOT BE IMPROVED BY ADDING DERIVED SERIES.
  It responded to more REAL series (Model170, +0.0069, 5/5) and to nothing else.
  The competition has no more labelled real series to give it.

One consistent detail worth keeping: fold 1 is hurt by EVERY augmentation
(-0.0110 re-cuts, -0.0053 re-cuts at 15%, -0.0013 splices, -0.0074 both,
-0.0070 mirrors). Fold 1 was already flagged as the anomalous fold years of
models ago in this project (students lose to Model84 there at t >= 128).
A budget artefact was ruled out: one epoch is one pass over every training
series and OneCycleLR's step count scales with the series count, so real series
still get their 24 passes; the augmented arms merely dilute the loss, which a
per-step tree tolerates and a sequence model does not.
Logs: models/model172/runs/train_synF.log.
