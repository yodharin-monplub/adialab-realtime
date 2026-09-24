Model141 - STAGE 5 of the user's protocol: the model (hyperparameters, seeds, ensembles).

Everything except the LightGBM parameters is Model111's (rows, soft targets,
40% splices, 317 columns). run_model141.py:
  search  : random search over leaves {15,31,63,127}, learning rate {.015,.03,.05},
            trees {600,1000,1600}, min_child_samples {100,300,1000}, colsample
            {.3,.45,.6,.8}, subsample {.6,.8,1}, lambda {0,3,10,30}, max_bin
            {63,127,255}; each config on pilot folds 0-1 (~10 min/config).
  confirm : best pilot config on folds 2-4.
  seeds   : 4 extra seeds of the current recipe on all folds (5-seed bag).
Ensembling across model families (LightGBM + XGBoost + ExtraTrees + GRU) is
evaluated on the saved OOF predictions with models/model112/multi_blend.py.
Status: written 04:22. Search (8 random configs, folds 0-1) ran 07:55-10:05,
gain vs Model111 pilot mean 0.62562 (fold 0 / fold 1 in brackets):
  cfg03 +0.0042 (-0.0007 / +0.0092)  leaves 31, lr .015, 600 trees, mcs 1000,
                                     colsample .8, subsample .8, lambda 10, bin 63
  cfg07 +0.0025 (-0.0016 / +0.0065)  leaves 15, lr .03, 600 trees, mcs 1000
  cfg01 +0.0018 (-0.0012 / +0.0048)  leaves 31, lr .03, 1600 trees, colsample .3
  cfg05 +0.0004, cfg04 +0.0003, cfg02 -0.0001, cfg06 -0.0021, cfg00 -0.0033
Pattern: less capacity (lr x trees = 9 instead of 30, 15-31 leaves, 1000 min
child) loses ~0.001 on fold 0 and gains 0.005-0.009 on fold 1.
Confirm cfg03 on folds 2-4 (16:07-16:47): -0.0039 +0.0003 +0.0013.
Five folds: -0.0007 +0.0092 -0.0039 +0.0003 +0.0013, mean +0.0012,
s.e. 0.0022, wins 3/5 -> does NOT pass the rule (fold 1 was the usual
outlier). Hyperparameters stay as they are.
Seeds (23:37-01:14, seed offsets 1000-4000 on the LightGBM random_state only;
bag_eval.py, logit mean, held-fold TS-AUC; score.txt):
  1 seed  0.6234 (reference a050)      4 seeds 0.6239 (+0.0005)
  2 seeds 0.6237 (+0.0003)             5 seeds 0.6240 (+0.0006)
  3 seeds 0.6239 (+0.0005)
Single seeds vary +/-0.003 per fold around the reference, but averaging
buys only +0.0006 - the student's error is bias (features), not variance.
runs/bag5_fold{k}.npz = 5-seed bag probabilities for the ensemble step.
Ensemble (01:28-01:38, ensemble.py -> ensemble.log): chain + w_i z(logit)
over bag5, gru (mean of the Model124 GRU sets, runs/gru_fold{k}.npz), E
(Model140), cfg03, splx8k; w in {0,.25,.5}; gain vs the chain alone:
  Submission31 structure (d111 .5 + gru .5)   +0.0034 (dev, 5-fold mean)
  best of 243: gru .5 + E .5                  +0.0044   (= holdout-tested
                                               candidate, +0.0002 there)
  pilot-best: gru .5 E .5 cfg03 .25 splx8k .25  pilot +0.0053, confirm +0.0022
bag5, cfg03 and splx8k get weight 0 in every top row: they are the same
model family and add nothing once E and the GRU are in. Stage 5 closed.
