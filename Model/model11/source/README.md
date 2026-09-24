# Structural break real-time experiments

The existing Models 1–10 and their saved results are preserved. Models 11–16
extend them with compact causal features, CPU model diversity, a time-matched
pairwise objective, time-varying stacking and residual boosting. Every numbered model has a
`readme.txt` alongside its detailed `README.md`.

Run from this directory:

```bash
python3 models/download_training.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python3 models/run_next_models.py run
OPENBLAS_NUM_THREADS=1 python3 models/model14/run_model14.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python3 models/model15/run_model15.py
OPENBLAS_NUM_THREADS=1 python3 models/model16/run_model16.py
OPENBLAS_NUM_THREADS=1 python3 models/report_next_models.py --bootstrap 200
python3 tests/test_compact_models.py
OPENBLAS_NUM_THREADS=1 python3 models/validate_compact_serving.py
python3 models/snapshot_compact_models.py check
```

The downloader uses `CRUNCH_API_KEY`, falling back to the existing `AGENTS.md`
credential. It downloads only the official training inputs. No key is copied
into model source, model artifacts or source snapshots.

The new experiments restore unlabeled row metadata, verify the saved Model 6
score, and reuse the original 8,333 development series and five complete-series
folds. They exclude outer-holdout and test labels. The shared 154-column cache
uses about 2.4 GiB; models run on four CPU threads. ExtraTrees is the slowest
candidate. Completed folds can be resumed with the same command.

Results are in `models/NEXT_RESULTS.md` and `models/next_models_results.json`
after reporting completes. Per-model protocols, fitted fold models and OOF
predictions are under `models/model<N>/runs/`. All reported improvements are
local development evidence. Blends and stacks inherit adaptive upstream model
selection; conditional bootstrap intervals do not make those scores independent
confirmation. No competition submission is performed by these commands.

Models 11–13 include Crunch train/infer entry points; they must be packaged
with the shared modules. Models 14–16 remain research stacks/blends. Model 6 stays the
deployment baseline pending stronger validation of a candidate.

The copied historical workspace lacks the large original detector feature
matrices and several canonical root modules. The new workflow needs neither.
Historical source snapshots remain available, but the original full training
pipelines still require their documented missing caches/dependencies.

The scoring implementation is checked against the [official time-stratified
scorer](https://github.com/crunchdao/competitions/blob/master/competitions/structural-break-real-time/scoring/scoring.py).
