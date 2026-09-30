TRAINING CODE of this submission (the bundle itself only runs inference).

All models in resources/ were trained offline on the 10,000 public training series of the
competition (8,333 development series in 5 series-level folds + 1,667 series sealed as an outer
holdout that was used only for go/no-go checks). No external data is used by any model in the bundle.
The files keep their original relative paths (models/modelN/...), because the pipeline verifies its
own source files by SHA-256 and path. PIPELINE_SUBMISSION31.txt explains the prediction step by step.

Order in which the code is run (each step writes caches that the next one reads):
 1. models/model5/model5_pipeline.py, models/model10/sources.py
        series-level folds, sealed holdout, development row layout (one row per online step).
 2. models/model20/features.py, run_model20.py (+ model18/augmentation.py, model21)
        causal detector features (the feature_pipeline.py of the bundle, 31 detector banks),
        splice augmentation (history + pre-break of series A, real post-break tail of series B).
 3. models/model57/whitening.py, run_model57.py
        history-only AR / EWMA / empirical-CDF whitening features (61 columns).
 4. models/model62/run_model62.py, models/model26/run_model26.py
        first-level voters (LightGBM / XGBoost / forests per detector bank), out-of-fold stacking
        (logistic regression on 34 logits), rolling-max-128 and the compact model = the "chain".
 5. models/model84/run_model84.py
        the frozen "expressive" LightGBM recipe (317 columns, 400k real + 400k splice rows).
 6. models/model109/teacher_features.py, run_model109.py, models/model111/run_model111.py
        oracle distillation: a teacher with future-seeing features, cross-fitted; the causal student
        is trained on 0.5*label + 0.5*teacher probability.
 7. models/model124/export_data.py, train_gru.py, select_gru.py
        causal GRU students on the same soft targets (15 networks, numpy inference in the bundle).
 8. models/model144/split_tests.py, run_model144.py
        27 split-test features: likelihood-ratio scan for the break position on AR-whitened values,
        then two-sample tests between the clean reference and the estimated post-break segment.
 9. models/model147/build_recuts.py, run_model147.py
        "move the start" re-cuts of the training series as extra training rows; the final student
        (arm named in resources/manifest.json["model147_student"]["arm"]).
10. models/model112/select_blend.py, models/model147/holdout_eval147.py
        blend constants (z-scores) and the sealed-holdout check of the final candidate.
Metric: tsauc.py (identical to the official TS-AUC). Every fold model is shipped; the bundle
averages the five fold voters of each model, nothing is refitted on all data.
