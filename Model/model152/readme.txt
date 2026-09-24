Model152 - is the student series-limited? -> full-development refit (2026-09-21).

1. LEARNING CURVE (run_model152.py, arm LC2, lc.log)
   SRE40 recipe, but each held fold also loses one fitting fold (3 -> 2 of the
   5 dev folds, ~5,000 -> ~3,333 series; real rows, teacher targets, re-cuts
   of that fold dropped; splices kept).
     fold 0 -0.0051   fold 1 -0.0045   fold 2 +0.0020   fold 3 -0.0073
     fold 4 -0.0043   mean -0.0039, s.e. 0.0016, worse on 4/5
   -> still clearly series-limited. Log-linear extrapolation of one voter to
   2x the series: about +0.006 (the deployed 5-voter average already recovers
   part of that, so the real gain must be measured).

2. HOLDOUT COMPONENTS (holdout_components.py -> holdout/, ~1.8 GB)
   One streaming pass over the 1,667 sealed holdout series with the
   Submission31 bundle that SAVES, per row: chain margin, GRU logit and the
   527 student inputs (439 detector union + 61 whitening + 27 split tests).
   offline_holdout.py then scores ANY student voter set on the holdout in
   minutes.  `offline_holdout.py check`: reproduces the BAG3 holdout run to
   6e-8 (smoke, 8 series).
     python models/model152/offline_holdout.py score TAG constants_m147_bag3.json model147:SRE40,SRE40b,SRE40c

3. FULL-DEV REFIT (run_full.py, full.log)
   teachers: 5 new cross-fitted teachers (each on 4 dev folds) give the soft
   target of every dev row.  Voter v = SRE40 recipe with fold v's columns and
   seed on ALL dev series (667k sampled rows, fold v's splices, all re-cuts).
   Arms FULL / FULLb / FULLc (re-cut sets "", _s148, _s149).
     python models/model152/run_full.py teachers
     python models/model152/run_full.py FULL
   Judged on the sealed holdout vs the fold-voter versions (SRE40, BAG3).
   If it wins, the final bundle refits on all 10,000 series (dev + holdout).

RESULTS 2026-09-21 (sealed holdout, 1,667 series, offline_holdout.py; parity
with the BAG3 holdout run 8.9e-8 on all 840,797 rows)
  new teachers (4 dev folds each): TS-AUC 0.732 0.699 0.717 0.697 0.726
  (old 2-fold teachers 0.686-0.704)
                         student raw   final blend   vs Sub33/BAG3
  SRE40 fold voters        0.62702       0.63376       -0.0007
  FULL  (set "")           0.63175       0.63513       +0.0007
  FULLb (_s148)            0.63441       0.63585       +0.0014 (P>0 0.97)
  BAG3  (Sub33)               -          0.63448          0
  BAGF3 = FULL+b+c         0.63416       0.63578       +0.0013
    bootstrap: sd 0.0005, 95% [+0.0005, +0.0023], P(gain>0) 1.00
    halves +0.0020 / +0.0005; steps 0-64 +0.0018, 64-256 +0.0012,
    256-512 +0.0012, 512+ +0.0017
  (BAGF3 uses BAG3's blend constants constants_m147_bag3.json; nothing tuned
  on the holdout.)
Sub33 cloud (run 119142): public 0.63311, rank 102/748 - holdout->cloud -0.0014.
Emailed the user (email_progress.txt). Not uploaded.
Next: full-dev refit of the GRU component; final refit on dev + holdout.
