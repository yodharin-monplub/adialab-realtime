Model161 - GRU students on the oracle-teacher targets (2026-09-23).

The deployed GRUs (weight 0.5 in the blend) learned 0.5*y + 0.5*old teacher.
Better targets transferred for the tree student (Model157/158), so:
soft_b.py re-fits Model157's teacherB per (held fold k, fold j) and scores ALL
rows of fold j (teacher TS-AUC 0.69-0.73) -> data/softB.npy; train_gru_b.py =
Model124's training loop unchanged with these targets (h96, 3 layers, 24 ep).
Best held-fold TS-AUC vs the deployed h96 L3 fold voters:
  fold 0  0.6352 vs 0.6354      fold 1  0.5977 vs 0.6020
-> no gain on 2 folds; stopped (CPU ~2x slower due to OneDrive sync).
The GRU part is limited by seed variance, not by its targets (cf. Model153).
CLOSED.
