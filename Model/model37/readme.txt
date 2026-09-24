Model37: independently seeded bag of the Model26 depth-4 voters.

Rationale
---------
Model26 is the strongest statistically accepted local model. Later feature,
meta-model, synthetic-data, and residual-correction experiments have not
produced a stable improvement. Model37 therefore tests variance reduction:
it refits the aligned-25 and aligned-60 depth-4 voters with independent XGBoost
random seeds while holding folds, fitting rows, synthetic examples, weights,
features, hyperparameters, and ensemble logic fixed.

The primary candidate averages the original Model26 probability with the
independently seeded replica probability before applying the frozen Model20
stack-column replacement and post-processing. This is a predeclared 50/50 seed
bag, not a validation-tuned blend.

CPU only: four XGBoost threads. No cloud submission is performed by this model.

