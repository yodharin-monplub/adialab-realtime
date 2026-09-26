Model176 - the 2025 challenge's REAL labelled series, given to the neural student
(2026-09-26).

WHY
The nets are the one component that responded to more REAL series (Model170:
+0.0069, 5/5, from the 1,667 holdout series) and they rejected every DERIVED
series (Model172/173: re-cuts, splices, mirrors). The 2025 Structural Break
Challenge data is the only other source of real labelled series; the rules allow
it ("Yes, but it is not officially encouraged"). It had never been given to the
nets - it failed in Model145 only as TREE rows, in an older recipe.

DATA - build_ext_sequences.py
models/model145/convert_2025.py's conversion reproduced draw for draw (seed 145):
each 2025 series (before B, after A) becomes history = B[:-pre], online = B[-pre:]
+ A[:post] with the online length and break position drawn from THIS year's
training distributions, then built into full per-step sequences with Model172's
machinery (102 contract inputs, 0.5*y + 0.5*oracle-teacher soft targets).
  7,232 series converted (2,769 dropped: history would fall under 1,000 points),
  3,066,157 steps, break share 0.293, positive steps 0.144, ~625 MB.
  None is in any development fold -> row_fold = -1 (every fold's net may use them,
  no leak); teacher j = id % 5 (no teacher ever saw these series).
Trained with models/model172/train_gpu2.py on a rented RTX 3090 (Taiwan host,
chosen for the upload path from Thailand), 24 epochs, hidden 256, seed 0:
  --data sequences_all.npz,seq_ext.npz --synth-share {0.4 | 0.2}

RESULT - NETS ALONE (held dev fold, vs Model170's nets = the ones in Sub42)
  fold      Model170    ext40      ext20
    0       0.637716   0.635777   0.638884
    1       0.610761   0.611442   0.612949
    2       0.631663   0.629002   0.634867
    3       0.609817   0.610699   0.609142
    4       0.629898   0.634648   0.634025
  ext40  +0.0003, 3/5 -> fails
  ext20  +0.0020 +/- 0.0008, 4/5, 2.4 s.e. -> PASSES the project rule.
  THE FIRST ARM TO PASS in two days of experiments. The lighter dose wins, as
  expected for series from a different generator.

RESULT - IN THE BLEND (the number that decides deployment)
  pilot-selected weights: Sub42's nets 0.64325 (ws 1.5, wg 1.0) vs ext20 nets
    0.64356 (ws 1.0, wg 1.0) = +0.0003; pilot folds +0.0007, confirmation +0.00005
  at Sub42's FIXED weights (only the nets change):
    per fold +0.00116 +0.00009 -0.00003 -0.00165 +0.00172
    mean +0.00026 +/- 0.00058, wins 3/5  -> NOT significant
  The raw net gain passes through to the blend at about 1/6, exactly the ratio
  seen for Model170 (raw +0.0069 -> blend +0.0013 -> public +0.0004), which would
  predict roughly +0.0001 on the public score.

CONTROL - models/model172/train_gpu2.py with sequences_all ONLY, seed 0:
  0.637716 0.610761 0.631663 0.609817 0.629898 - identical to Model170 to six
  decimals. The baseline is valid, and a run with a given seed is deterministic.

SECOND SEED - the result REVERSES
  ext20 seed 0   +0.0012 +0.0022 +0.0032 -0.0007 +0.0041  mean +0.00200  4/5
  ext20 seed 1   -0.0036 -0.0024 +0.0007 -0.0034 -0.0005  mean -0.00185  1/5
  two-seed avg                                            mean +0.00007  2/5
  seed-to-seed swing per fold: 0.0048 0.0046 0.0025 0.0027 0.0047 (mean 0.0039)
  -> A single net's score moves by ~0.004 per fold with its seed - more than any
  net effect measured in Model170-176. Every single-seed net comparison in this
  project should be read with that in mind.
  The swing has the SAME SIGN on all five folds. Reason: seed 0 shares its
  initialisation (torch.manual_seed(1000 + 17k + seed)) and batch order with the
  seed-0 baseline, so ext20-seed0 vs baseline is a PAIRED test where only the data
  differs; ext20-seed1 vs baseline changes the initialisation too. The clean second
  replicate is therefore ext20-seed1 vs a CONTROL at seed 1 (ctrls1_), running now.

PAIRED CONTROL AT SEED 1 (ctrls1_: sequences_all only, seed 1)
  0.638677 0.604320 0.635865 0.605290 0.627866
  paired seed 0 (ext20 - ctrl, same init)  +0.0012 +0.0022 +0.0032 -0.0007 +0.0041
                                           mean +0.00200 +/- 0.00083, 4/5
  paired seed 1 (ext20 - ctrl, same init)  -0.0046 +0.0040 -0.0035 +0.0012 +0.0015
                                           mean -0.00029 +/- 0.00162, 3/5
  POOLED over both seeds                   mean +0.00086 +/- 0.00092, 7 of 10
                                           fold-differences positive = 0.9 s.e.
  init noise alone (ctrl s1 - ctrl s0, IDENTICAL data):
                                           +0.0010 -0.0064 +0.0042 -0.0045 -0.0020
-> CLOSED. The seed-0 "pass" was one favourable initialisation. Pooled over two
paired initialisations the effect is +0.0009 at 0.9 s.e.; even if real, the ~1/6
pass-through from nets to blend would leave about +0.00015 - invisible on any
leaderboard and far inside the private-set noise. Not built, not uploaded.

THE LESSON THAT OUTLIVES THIS MODEL
Two nets trained on IDENTICAL data differ by up to 0.0064 on a single held fold
purely from their initialisation. That is larger than every net effect measured
from Model170 to Model176. Single-seed net comparisons in this project are
therefore not trustworthy on their own; a net change needs PAIRED seeds (same
initialisation with and without the change), at least two of them, before it is
believed. Model170's +0.0069 is large enough to survive this; nothing since is.
(Model169 had already shown that averaging more seeds per fold does not raise the
sealed-holdout blend, so this is noise to design around, not a lever.)

Cost of the whole Model176 run: ~2 h on a rented RTX 3090 ($0.24/h), instance
destroyed afterwards. Logs: runs/train_ext.log, train_ctrl.log, train_ctrls1.log.
