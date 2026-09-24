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

RESULT (vast.ai RTX 3090 $0.158/h, 10 runs in ~35 min, instance destroyed;
the first host sat in "loading" for 25 min and was swapped out)
  held development fold, dev-only (Model169) -> all-series (this model):
    fold 0  0.636866 -> 0.637716
    fold 1  0.595178 -> 0.610761
    fold 2  0.628359 -> 0.631663
    fold 3  0.604497 -> 0.609817
    fold 4  0.620336 -> 0.629898
    mean    0.617047 -> 0.623971   (+0.0069, 5/5)
  hidden 384 all-series: mean 0.622368 -> no better than 256.
-> the neural part was starved of series, not of capacity or targets.
Model171 weight protocol with these nets: dev blend mean 0.64276 (deployed
nets 0.64149, Model169 nets 0.64119); pilot-selected ws=1.5 wg=1.0
(pilot +0.00089, confirm +0.00023); ws=1.0/wg=0.75 was the most robust
alternative (+0.00082/+0.00044, 5/5).
SUBMISSION42 = Submission41 with these 5 nets and weight_gru 1.0
(models/model169/deployment/build41.py with SOURCE/NETSDIR/GRUTAG/WGRU,
validate41.py with the new logits: parity 1.2e-7, replay 0.0, runtime +1.6%,
~4.8 cloud hours; sealed as submissions/submission42.zip).
CAVEAT: these nets trained on the holdout series, so the sealed holdout can no
longer judge them; the evidence is the 5/5 development-fold win.
Emailed the user for approval; NOT uploaded.
