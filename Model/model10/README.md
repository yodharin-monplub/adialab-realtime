# Model 10: Causal Post-processing

Model 10 tests whether a small, streaming score-history correction improves
TSAUC over frozen out-of-fold predictions. It changes no detector or upstream
model. The primary source is Model 6 after its frozen global decay, so every
candidate must beat the actual incumbent. An optional secondary source is
Model 9's exploratory pairwise stack; results from that source are not
deployable evidence.

The completed experiment outcome is recorded in [`RESULTS.md`](RESULTS.md).

## Candidate families

The frozen grid contains raw output as the fallback plus 127 candidates in four
strictly causal families:

- **Logit EMA trend:** add a small signed deviation from an exponentially
  weighted logit state.
- **Rolling-maximum premium:** blend the current logit toward a finite-window
  maximum; old peaks expire.
- **Soft decayed peak:** maintain
  `peak_t = max(p_t, decay * peak_(t-1))`, then blend only 5–35% toward its
  logit. Decay ranges from 0.50 through 1.00.
- **Prior EW innovation:** add a clipped, self-normalized innovation computed
  against the state at `t-1`, then update that state.

Batch and one-row streaming implementations share the same transform contract
in `causal_transforms.py`.

## Conditional nested validation

Model 10 reads development OOF predictions only. Each of the five original OOF
folds is independently divided into five complete-series calibration cells.
For each held cell, the transform is chosen using only its other four cells.
Its TSAUC is then measured on that untouched cell. Cell gains are pair-count
weighted inside each original fold before the five-fold paired SE and promotion
rule are computed. The unmodified incumbent is retained unless the paired mean
gain is at least `0.00025` and `mean_gain - 0.5 * SE` is at least `0.00010`.
The pipeline also assembles
all cross-fitted predictions, but labels that full-fold score diagnostic-only:
different cells can select different transforms, making their margins
heterogeneous in cross-cell rank comparisons.

Stage-one promotion additionally requires improvement on at least four of five
original folds and a non-raw all-development candidate satisfying the frozen
gate. Even then, deployment remains blocked until a confirmatory paired
complete-series bootstrap is run. The current pipeline records that requirement
but does not auto-promote a transform.

This is conditional evidence: upstream detector, feature, and model selection
already used the development folds. The label loader filters the series-level
index to development IDs before materializing `tau_index`; it never loads the
full target array or retains a sealed-series label. No command scores the
sealed outer holdout, reduced test labels, or competition test data.

## Run

From the repository root:

```bash
python3 models/model10/run_model10.py status
python3 models/model10/run_model10.py run --model6-only
python3 models/model10/run_model10.py run
```

The second command evaluates the deployable Model 6 source only; the third also
evaluates Model 9. Runs are checkpointed under the configuration-derived
`models/model10/runs/postprocess_*/` directory. `summary.json` and `RESULTS.md`
are written only after all requested sources finish. The notebook provides the
same status and run entry points, with execution disabled by default.

## Limitations

- Candidate rankings across the entire development set are diagnostics, not
  confirmatory estimates.
- The Model 9 source inherits its not-fully-nested upstream stack limitation.
- A passing result still needs streaming integration, parity tests, runtime
  measurement, and the independent bootstrap gate before submission work.
- Globally monotone calibration is omitted because it cannot change TSAUC's
  within-time rankings.

## Complete source copy

[`source/`](source/) contains a hash-manifested mini-repository with Model 10,
its upstream model packages, shared pipelines, imported detectors, and tests.
Generated checkpoints, predictions, caches, and training data are excluded.
