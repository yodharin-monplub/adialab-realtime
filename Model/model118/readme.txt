Model118: higher-capacity distilled student.

Hypothesis: soft targets carry far less label noise than 0/1 labels, so the
student can use more capacity than Model84's expressive recipe (63 leaves,
1000 trees, lr 0.03) without over-fitting. Model118 uses 127 leaves, 2000
trees, lr 0.02, min_child_samples 200; fit rows, 40% aligned synthetic mass,
Model111's cached cross-fitted teacher probabilities, 317 fold columns and
weights are unchanged. The recipe override is applied only after the frozen
Model84 protocol check passes. Raw held-fold component reported; the
top-level blend is evaluated with models/model112/multi_blend.py.
Baseline: Model111 raw components 0.644007/0.607235/0.627220/0.608750/0.629653.

Run: .venv\Scripts\python.exe -u models\model118\run_model118.py 0 1
Pilot (raw held-fold TS-AUC)
fold      Model118   Model111
0         0.642077   0.644007   (-0.0019, 662 s fit)
1         0.600118   0.607235   (-0.0071, 660 s fit)

Result: more capacity hurts on both pilot folds. Soft targets do not make the
student tolerate 127-leaf/2000-tree trees; Model84's 63-leaf/1000-tree recipe
remains the right size. Confirmation folds not run. Closed.
