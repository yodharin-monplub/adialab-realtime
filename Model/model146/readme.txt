Model146 - the student trained as a RANKER with query = online step t
(from the user's question 2026-09-20 "is there other way to create the series?",
option C: same rows, different grouping).

TS-AUC compares series with each other at the same step and averages over
steps. The current student is a cross-entropy classifier whose row weights
imitate that. Here nothing changes except the objective: LightGBM LGBMRanker
(lambdarank / rank_xendcg), rows sorted into one query per step, labels = the
soft target 0.5*y + 0.5*teacher binned to integers 0..8 (ranking objectives
need integer gains), same 317 columns, same 400k real + 400k splice rows,
same weights, same trees/leaves/learning rate.
Evaluation: held-fold TS-AUC of the raw ranking score vs Model111 (a ranker's
score is only meaningful within a query, which is exactly what TS-AUC uses;
for blending it is z-scored like every other component).
Rule: gain > 2 s.e. and >= 4/5 folds, then the sealed holdout.
RESULT 2026-09-20 (fold 0, gain vs the classifier student 0.644007):
  rank_xendcg                      0.614637  (-0.0294)   286 s
  lambdarank, truncation 300       0.625908  (-0.0181)  1219 s
  lambdarank, full-query truncation: > 35 min per fold, stopped unfinished.
Both far worse; fold 1 not run. Reasons: the NDCG-type objectives care about the
top of each query while AUC counts every pair; every query gets equal say while
TS-AUC weights a step by n_pos*n_neg; and the soft teacher target has to be
binned to 9 integer grades. The cross-entropy student with metric-shaped row
weights stays. Closed.
