Model33 — nonlinear meta-learner over the full primitive OOF bank

Research basis
--------------
A public high-ranking ADIA Structural Break solution uses multiple diverse base
learners and a small XGBoost meta-learner over base logits plus their mean and
dispersion. Our Model9 already stores 34 diverse primitive OOF predictors, but
combines them with a strongly regularized linear pairwise stack. Model33 tests
that missing nonlinear level-two architecture without adding another correlated
detector bank.

Frozen inputs
-------------
For every row, the level-two matrix contains all 34 Model9 primitive OOF logits,
their mean, standard deviation, minimum and maximum, Model26's final ensemble
margin, Model27's raw matched-context logit, Model30's raw matched/no-context
logit, log(1 + online time), and the 13 clean-history descriptors defined by
Model27. The source manifest and every upstream protocol are checksum-verified.

Validation and model
--------------------
For held fold f, fitting_rows() supplies 400,000 rows from three other folds and
excludes both f and adjacent stopping fold (f+1) mod 5. One fixed XGBoost model
uses depth 2, 400 rounds, learning rate .03, min_child_weight 50, subsample .8,
colsample_bytree .8, L2 20, L1 .1, hist trees, four CPU threads, and seed
330911+f. Weights are balanced by online time and target. The held fold is never
used for fitting or standardization. Output probability is scored directly.

Promotion requires mean held-series OOF TSAUC >= .640, gain >= .0007 versus
Model26, at least 4/5 fold wins, positive whole-series bootstrap lower bound,
and positive gain in both fixed ID-hash halves.

The primitive OOF bank has the already-documented Model9 level-two nesting
caveat: a training-row base predictor can have been trained on another meta-held
fold. Results are therefore exploratory rather than a pristine nested estimate.
No outer holdout or competition labels are read. CPU only; no automatic upload.
