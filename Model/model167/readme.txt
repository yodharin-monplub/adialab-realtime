Model167 - "stitch" rows: the re-cut trick applied to the spliced series
(2026-09-23).

build_stitch.py re-assembles each fold plan's 1,000 splices (aligned, exactly
as deployed) and re-cuts each one (s uniform in [8, tau-8]), so glued series
with real post-break tails get their break at a different step. 50 sampled
rows per series with the 500 union columns, 27 split tests and 32 teacher
inputs -> data/stitch_fold{k}.npz (~50k rows per fold, ~14 min/fold at 2
workers; Windows Smart App Control intermittently blocks library DLLs in
worker processes, fewer workers avoids it).
holdout_stitch.py: Submission40 recipe on development series only, stitch rows
added to the non-real block with oracle-teacher soft targets.
  SEALED HOLDOUT        student raw   final ws=0.5   final ws=1.5
    without stitch        0.640603      0.637265       0.640380
    with stitch           0.641284      0.637258       0.640474
-> the student gains +0.0007 raw but the blend gains +0.0001: not worth the
extra data path in the bundle. CLOSED (data kept).
