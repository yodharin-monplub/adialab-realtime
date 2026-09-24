# Submission 6: Timeout-Safe Model 6 Ensemble

This is the push-ready, timeout-safe deployment of Model 6's frozen candidates.
It uses only models fitted from the updated 10,000-series public `train/` data. No
competition test values, holdout labels, OOF arrays, rejected candidates, or
training caches are bundled.

## Reported validation result

- Leakage-controlled sealed holdout TSAUC: **0.626144**
- Complete-series bootstrap standard error: **0.011365**
- Bootstrap 95% interval: **[0.603879, 0.649604]**
- Blend-tuning TSAUC: **0.626680**
- Frozen experiment contract: `92f5511cd65d6432bd35bcb2d8928fdd4cf2c1cea421ebc7299063db96ef7742`

The holdout was not used for detector, feature, model, hyperparameter, blend,
decay, or runtime-profile selection. The six-component recipe and removal of
the runtime-heavy final ExtraTrees component were fixed from development data
before the revised holdout score was calculated. This estimate is local
validation, not a guarantee of the private competition score.

## Runtime architecture

For each newly revealed value, `CoreStreamDetector` updates the union of 20
core and 11 candidate causal feature banks. Their 155 detector voters contain
10,397 trees and are traversed in one native LightGBM call. The retained global
recipe blends the core detector mean, candidate ExtraTrees, core XGBoost,
candidate LightGBM meta-model, core linear meta-model, and candidate random
forest. Component persistence and the final causal rule
`state_t = max(raw_t, 0.95 * state_(t-1))` exactly follow the frozen recipe.

The original seven-component profile included a 5.6-million-node deep
ExtraTrees bag for only `0.000045` development TSAUC. Removing it reduced the
bundle to about 94 MiB and the streaming benchmark to about 10.5 ms per point.
More importantly, `train()` now performs only static manifest/hash validation:
LightGBM, XGBoost, the feature pipeline, and fitted models are first loaded
inside each inference worker. This avoids the OpenMP-after-fork deadlock that
caused the cloud timeout. The full 50,983-point Crunch test completed in
**4 minutes 35 seconds** with four workers and about **298 MB** added memory.
State is isolated per series, one probability is emitted per yield, and future
values are never inspected.

## Files

- `main.py`: Crunch entry points and four-worker streaming contract.
- `solution.py`: loading, validation, streaming inference, and blending.
- `feature_pipeline.py`, `detector/`: exact audited causal feature code.
- `compact_runtime.py`: pickle-free forest and linear-meta inference.
- `resources/`: pretrained frozen voters and signed model manifest.
- `build_resources.py`: reproducible local exporter; it is not called by Crunch.
- `validate_bundle.py`: force-first-train and deterministic streaming smoke test.

`train()` intentionally validates and preserves the uploaded frozen resources;
it does not refit models in the cloud or initialize worker-facing ML libraries.
Resource SHA-256 hashes are checked on every training-stage validation.

See `SUBMIT.md` for the local test and push commands.
