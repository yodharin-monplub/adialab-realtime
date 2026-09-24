Model113: second oracle-distilled voter (alpha = 0.25) for variance reduction.

Identical to Model111 (frozen Model84 expressive recipe, same fit rows, 40%
aligned synthetic mass, 317 fold columns) except real-row targets are
0.25*y + 0.75*teacher. Model111's cached cross-fitted teacher probabilities for
the same fit rows are reused (copied to runs/), no teacher refit.

Run: .venv\Scripts\python.exe -u models\model113\run_model113.py 0 1 2 3 4

Raw held-fold component TS-AUC
fold      0         1         2         3         4         mean
M113    0.642853  0.607750  0.625754  0.608527  0.626032  0.622183
M111    0.644007  0.607235  0.627220  0.608750  0.629653  0.623373
M84     0.643302  0.599595  0.621639  0.605160  0.627773  0.619494

Top-level blend on the Submission28 chain (models/model112/multi_blend.py d111 d113):
adding Model113 to Model111 does not help (pilot-best d111=0.25,d113=0.25 gives
+0.00145 overall vs +0.00147 for Model111 alone at 0.5). Not used in Submission30.
