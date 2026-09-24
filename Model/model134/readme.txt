Model134: block-bootstrap replicas as synthetic training data (user suggestion, 00:08).

Why: the only synthetic data so far are splices (prefix of series A + post-
break tail of series B), which teachers/GRUs find far easier than real breaks
(teacher AUC 0.91 vs 0.70). A moving-block bootstrap of each real series keeps
the true break magnitude and dependence structure while producing a new sample
path, i.e. realistic difficulty.

build_bootstrap.py: for every development series, blocks (length 48) are
resampled from the pre-break part (history + online[:tau]) to form a new
history (same length) and pre-break online segment (same tau), and from the
post-break part for the rest; break-free series are bootstrapped whole; the
replica is re-standardized by its new history. 80 online rows per replica get
the full causal features (detector union of the five fold contracts + 61
whitening columns via the frozen pipeline). One replica per series, tagged by
donor fold for fold-clean use.

run_model134.py: Model111 recipe; arms "replace" (bootstrap rows instead of
splices, 40% mass) and "add" (splices 40% + bootstrap 25% of the remaining
real mass). Pilot folds 0-1 vs Model111 (0.644007 / 0.607235).
Dataset: 8,333 replicas, 644,592 rows x 500 columns, built in 55 min (6 workers).
Pilot (raw held-fold TS-AUC of the distilled student)
arm "replace" (bootstrap instead of splices, 40% mass):
  fold 0  0.644160 vs 0.644007 (+0.0002)
  fold 1  0.585262 vs 0.607235 (-0.0220)  <- fold 1 collapses without the
          splices, exactly as Model109 (no synthetic rows) did: fold 1's breaks
          are learned from the spliced examples, not from real-like replicas.
arm "add" (splices 40% + bootstrap 25% of the remaining real mass):
  fold 0  0.642495 vs 0.644007 (-0.0015)
  fold 1  0.602142 vs 0.607235 (-0.0051)
Conclusion: block-bootstrap replicas of real series neither replace nor
complement the splices; they dilute the training mass with examples that
carry no new break information (the tree already sees the donors' own rows
with soft targets). The splices' value is precisely that they create breaks
that do not exist in the real data (new A/B combinations). Closed 01:30.
