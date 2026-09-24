Model114: oracle-distilled version of Model64's 500-feature LightGBM voter.

Model64 recipe (439 detector + 61 whitening columns, 31 leaves, 800 trees,
lr 0.035, 40% aligned synthetic mass; synthetic 500-column matrix = Model20
256-column cache + Model64 183-column "missing" cache + Model57 whitening),
same frozen fit rows. Real-row targets 0.5*y + 0.5*teacher using Model111's
cached cross-fitted teacher probabilities (identical fit rows, asserted).

Run: .venv\Scripts\python.exe -u models\model114\run_model114.py 0 1 2 3 4

Raw held-fold TS-AUC
fold       0         1         2         3         4         mean
M114     0.640052  0.605522  0.623078  0.611173  0.628827  0.621730
M64      0.638220  0.602117  0.622235  0.608485  0.631730  0.620557
gain     +0.00183  +0.00341  +0.00084  +0.00269  -0.00290  +0.00117 (4/5 wins)

Top-level (models/model112/multi_blend.py d111 d114): adding Model114 to the
Submission30 blend does not help (pilot-best keeps d114 = 0). Not deployed.
