Model 18 — fold-local splice augmentation (CPU only)
==================================================
Purpose: test whether joining a real prebreak prefix from A to a genuine
postbreak tail from B improves the standalone Model 11 compact LightGBM.
Model 16 and all cloud/deployment artifacts remain unchanged.

Run from repository root with the training environment:
  .venv/bin/python models/model18/run_model18.py run
  .venv/bin/python models/model18/run_model18.py report
The report command never retrains. The run command freezes all donor plans
before the first fit and resumes valid checkpoints. Checksummed corruption
raises an error; incomplete uncommitted artifacts can be recomputed.
Changed code/configuration/inputs/dependencies create a new run directory.

Fixed study: five development folds x five arms = 25 fits. Arms are a fresh
real-only control and raw/aligned splices at 10%/25% synthetic weight.
Each fold uses identical 500,000 real rows and Model 11 parameters (350
rounds, 15 leaves, learning rate .04, minimum leaf rows 150, L2 20,
feature fraction .9, max_bin 127, four CPU threads, seed 20260906+fold).
The fresh control must reproduce saved Model 11 scores within 1e-6.

Donors: only fitting-fold break series, no validation or reserved donors.
Up to 1,000 unique A donors/fold; A requires >=32 online prebreak and >=32
postbreak observations. Different B supplies enough contiguous genuine
postbreak values to preserve A's original horizon; B reuse <=5. No padding,
repetition or interpolation. Insufficient/nonfinite donors are rejected;
shortfalls are recorded. Same pairs and sampled rows across all four arms.
Aligned B offset = median(A_pre)-median(B_pre), using the last up to128
clean observations before each break, including history if needed. This
does not force boundary equality. Raw tails retain B's original values.
Assembled inputs are float32. Labels are zero before splice and one from
splice. These labels are assumptions: cross-series changes may be subtle
or unrealistic, and extra synthetic data need not improve real detection.

Original 154 causal features are recomputed on each complete composite
before selecting at most 500,000 synthetic rows/fold. Donor IDs, break
indices and final horizons are never model features. Augmentation seed is
20260918+fold, independent of control sampling; synthetic row sampling uses
SeedSequence([20260918+fold,1]). The real feature cache is reused read-only.

Weights: compute time_pair_weights on real fitting rows only. Within each
(online step,target) cell with synthetic coverage, move 10% or25% of the
original weight to equal-weight synthetic rows. Uncovered cells retain
their original real weights. Zero original mass => zero synthetic mass.
Cell totals and global weight scale are conserved; actual shares reported.
Synthetic rows also affect LightGBM bin construction and leaf row counts;
this study measures the complete fixed training recipe, not weights alone.

Evaluate only on complete, unmodified, held-fold real series with TSAUC.
Report paired fold gains and no-break series ever crossing .5 and .8.
Use 1,000 paired whole-series bootstrap replicates, stratified within the
original folds, seed180906, shared across all arms. Simultaneous one-sided
95% lower bounds use the centered bootstrap maximum of four contrasts.
Pass: mean gain >=.001, wins >=4/5, simultaneous lower bound >0. Rank passing
arms by gain, then lower augmentation share, then raw before aligned.
No adaptive grid expansion, ensembles, full-data training or submissions.

The reused development folds are not fresh confirmation. Bootstrap bounds
condition on saved predictions and omit upstream research/refit uncertainty.
The old outer holdout was historically revealed; it stays closed here.
No competition test labels are loaded. All outputs stay under Model18.
