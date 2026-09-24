Model155 - wider student inputs (2026-09-22).

Idea: each student voter sees only its fold's 317 columns (+ pruned split
tests). The cached union has 500 (the five fold selections) + 27 split tests;
the serving code maps voters by column name, so a wider voter would need no
serving change. Arm = SRE40 recipe with all 527 columns (run_model155.py).

Pilot U40 (splices had NaN outside their fold's 317 cached columns):
  fold 0 +0.0059   fold 1 -0.0118  -> confounded: NaN made splice rows
  recognisable, which changes how much the model trusts splices.
Clean arm U40C: build_splice_union.py computed all 500 columns for the
400k splice rows of every fold plan (parity with the cached 317 columns:
max rel 6.5e-8), same code as the re-cuts.
  U40C vs SRE40: fold 0 +0.0010  fold 1 -0.0024  fold 2 -0.0007
                 fold 3 -0.0005  fold 4 +0.0014   mean -0.0002, 2/5
-> no gain from wider inputs; the fold column selection is not the limit.
Side finding: the NaN pilot swung +0.006 / -0.012 between folds 0 and 1 -
the weight the model gives splices matters a lot and differently per fold
(as in stage 1). CLOSED.
Data kept: data/splice_union_fold{k}.npy (400k x 500 per fold, reusable).

