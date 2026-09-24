# Model 15: residual boosting

Instead of fitting a standalone classifier and blending it after training,
Model 15 starts from the Model 9 margin and learns a small conditional
correction. The 154 compact causal features and current Model 9 margin let
the correction depend on the series' historical properties and current evidence.

The fixed recipe uses 500,000 fitting rows from four complete-series folds,
150 LightGBM rounds, seven leaves, learning rate 0.025, minimum leaf size 1,000,
and L2 penalty 100. Training uses TSAUC pair weights and supplies the frozen
margin as LightGBM's `init_score`. Inference must explicitly add the correction's
raw score to the frozen margin before taking the logistic function.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python3 models/model15/run_model15.py
```

Five fold models, predictions and a source/data-hashed protocol are retained.
No held-fold label enters correction fitting; no sealed outer-holdout or test
label is loaded. The Model 9 inputs were not produced by base refits inside
each correction-held fold, so this remains exploratory stacking on repeatedly
reused development data. Streaming integration and independent validation are
required before any submission. No automatic deployment occurs.
