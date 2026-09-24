# Model 16: persistence plus feature diversity

This fixed blend combines the historical Model 10 rolling-maximum diagnostic
with Model 11's new compact signal. In margin space:

`score = 0.81 * model9 + 0.09 * rolling_max_128(model9) + 0.10 * model11`.

All components are causal within a single series. The runner first reproduces
Model 10's recorded diagnostic scores, then scores the fixed blend on the five
complete-series development folds. Its source-hashed protocol and OOF margins
are saved under `runs/`.

```bash
OPENBLAS_NUM_THREADS=1 python3 models/model16/run_model16.py
```

This is a post-hoc combination of diagnostics. Model 10's promotion gate failed,
and Model 9's upstream base models were not refitted inside each meta-held fold.
The same development data have been repeatedly used to choose models. Any
increase is exploratory; conditional bootstrap uncertainty does not include
adaptive selection or upstream fitting. No sealed holdout or test labels are
loaded, and no automatic deployment or prize claim is made.
