Model131: bidirectional-GRU teacher for the distilled tree student.

The Model109 teacher is a LightGBM on hand-made privileged features (LLRs
over candidate onsets); it reaches only ~0.70-0.73 TS-AUC even though it sees
the whole online segment. A bidirectional GRU over the same 102 causal inputs
sees the future directly and may localise breaks far better, giving sharper
soft targets. Model110/117 showed that modest teacher improvements did not
transfer; this tests a much larger teacher change.

Protocol (fold-clean as in Model111): teacher (k, j) trains on hard labels of
folds outside {k, k+1, j} and predicts fold-j rows -> soft targets for the
held-fold-k student (Model111 recipe, 0.5*y + 0.5*teacher on real rows,
hard labels on synthetic rows). 15 teachers on a vast.ai GPU
(train_teacher_bigru.py, remote_teacher.sh), then run_model131.py locally.
Gate: student raw >= +0.003 vs Model111 on pilot folds 0-1 before folds 2-4.

Teachers (vast.ai Quebec host 51318568, ~1 s/epoch, 15 models in ~6 min,
instance destroyed 21:56): fold-j TS-AUC 0.653-0.696 - WEAKER than the
Model109 tree teacher on hand-made privileged features (0.687-0.732). A
bidirectional GRU on causal window features does not localise breaks better
than explicit onset-likelihood features.
Student pilot (raw held-fold TS-AUC): fold 0 0.640460 vs Model111 0.644007
(-0.0035); fold 1 0.610069 vs 0.607235 (+0.0028). Mean -0.0004; the usual
fold split. A weaker teacher gives no better student. Closed 22:15.
Total vast.ai spend for this model about $0.5 (three dud hosts destroyed).
Confirmation folds: 2 0.625982 (vs 0.627220), 3 0.604620 (vs 0.608750),
4 0.627936 (vs 0.629653). Five-fold mean 0.62181 vs Model111 0.62337.
Bagging lead (logit average Model111 + Model131): 0.64405 / 0.61009 / 0.62812 /
0.60818 / 0.63079 = 0.62425, +0.00087 vs Model111, 4/5 folds - variance
reduction across teacher families (students correlate 0.97). Below the
"clearly higher" bar; kept as a documented lead for a future bundle.
