Model129: feature-level stacking - the GRU ensemble logit as a LightGBM column.

Motivation
Submission31 blends the distilled tree and the GRU ensemble at the score level
(+0.0016 holdout). A tree that SEES the GRU's per-step logit as a feature can
use it conditionally (by time, by regime) instead of with one global weight.

Stage 1 - decision pilot (pilot_leaky.py, fold 0 only)
Deliberately leaky: real training rows of fold j get the fold-j OOF GRU logit
(that GRU saw fold 0), synthetic rows get NaN. Compares the Model111 student
with vs without the column on held fold 0. Only a LARGE gain (>= +0.005 raw)
justifies the full build; small gains are indistinguishable from the leak.

Stage 2 - full build (only if stage 1 passes)
 a. Full-sequence inputs (61 whitening + 40 detector columns) for the 1,000
    spliced synthetic series of each fold plan (caches only hold sampled rows).
 b. 15 nested GRUs on a GPU: for held fold k and training fold j, train on
    folds outside {k, k+1, j}; predict fold-j rows, fold-k rows, and fold k's
    synthetic sequences. Fold-k labels never reach the GRU that scores fold k.
 c. Student = Model111 recipe + GRU-logit column(s), five folds; top-level
    blend on the Submission28 chain; bundle; parity; sealed-holdout check.
 d. Serving uses the full 4-fold GRU ensemble already in Submission31 (the
    usual nested-vs-full stacking mismatch; the holdout run measures it).

Stage 1 results (raw held-fold TS-AUC of the student)
fold 0: with GRU column 0.649341 vs Model111 0.644007 (+0.0053; GRU column
        gain share 9.1%).
fold 1: 0.598588 vs 0.607235 (-0.0086; gain share 9.0%).
fold 2: 0.625526 vs 0.627220 (-0.0017) although the GRU alone (0.6314) beats
        the tree on this fold - an informative column cannot make the tree
        worse unless its train/serve distribution is broken.
Inconclusive and dominated by the pilot's own defect: synthetic rows
(40% of the mass) carry NaN in the GRU column, so the tree learns "GRU present
=> real row" and misuses the column where the GRU is weak (fold 1).
Stage 1b (pilot_v2.py): synthetic rows get real GRU values - full-sequence
features for the spliced series (build_synthetic_sequences.py) scored by the
fold-k GRU ensemble on CPU (score_synthetic.py). Synthetic sequences: ~575k
rows per fold, built in ~3 min/fold; kept-row whitening columns match the
existing cache exactly (max diff 0).
fold 0 (v2): 0.644307 vs 0.644007 (+0.0003); GRU column gain share 18.7%;
        GRU logit mean on synthetic rows +0.29 vs -0.20 on real rows (splices
        are much more detectable, as the teacher's 0.907 AUC on them showed).
        The NaN pilot's +0.0053 was an artefact.
fold 1 (v2): 0.607636 vs 0.607235 (+0.0004); fold 2 (v2): 0.625079 vs
        0.627220 (-0.0021). Mean -0.0005.
CONCLUSION: feature-level stacking of the GRU into the distilled tree is no
better (slightly worse) than the linear score blend of Submission31; the
nested-GRU stage (2b/2c, scripts written: train_gru_nested.py,
remote_nested.sh, run_model129.py) is NOT run. Closed 2026-09-17 21:00.
