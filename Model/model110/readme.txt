Model110: Model109 distillation with a look-ahead teacher.

Same nested protocol, student rows, weights and settings as Model109. The
teacher additionally receives the top-60 causal features (Model109 fold-0 hard
student gain order) evaluated at future rows t+32, t+128 and the last online
row. Arms: mix (alpha 0.5) and mix25 (alpha 0.25).

Run: .venv\Scripts\python.exe -u models\model110\run_model110.py 0 1

Result (raw held-fold TS-AUC)
arm     fold0     fold1     mean
mix     0.643357  0.588782  0.616070
mix25   0.644396  0.588561  0.616479
Model109 mix (baseline teacher): 0.645190 / 0.589523 = 0.617357

Teacher TS-AUC improved slightly (fold0 0.700-0.728, fold1 0.719-0.739) but the
students did not improve. Conclusion: teacher strength beyond Model109 is not
the bottleneck; Model109's simpler teacher is kept. No submission.
