Model115: Model111 distillation with 3x more real rows (1.2M instead of 400k).

Real fit rows are a seeded sample of ALL rows in the three fitting folds
(the legacy 20% (series,step) hash sampling is dropped). Soft targets come
from Model111's saved cross-fitted teachers (teacher j scores the new rows of
fold j; folds k and k+1 never reach any teacher). Synthetic rows, 40% synthetic
cell mass, weights, 317 fold columns and the expressive LightGBM recipe
(63 leaves, 1000 trees) are unchanged.

Run: .venv\Scripts\python.exe -u models\model115\run_model115.py 0 1

Pilot (raw held-fold TS-AUC)
fold      Model115   Model111 (400k rows)
0         0.641638   0.644007   (-0.0024)
1         0.607174   0.607235   (-0.0001)

Result: no gain from 3x real rows (fold 0 worse, fold 1 flat). Confirmation
folds not run. Likely reasons: adjacent rows of a series are near-duplicates,
so the effective sample size barely grows while the tree budget (1000 trees,
lr 0.03) tuned for 800k rows now under-fits; the legacy 20% hash sampling also
acted as within-series decorrelation. Closed.
