Model 19 — augmentation continuation from the 0.622 Model 16 submission
====================================================================
User requested continuing the strong incumbent, not another weak standalone
model. Model 16 combines 90% of the persistence-corrected Model 9 backbone
with 10% of Model 11, in logit/margin space. This study keeps the backbone,
persistence and weights unchanged and swaps only that compact component
for the fold-local splice-trained models already fitted in Model 18.
No ExtraTrees experiment was implemented or started.

Commands from repository root:
  .venv/bin/python models/model19/run_model19.py run
  .venv/bin/python models/model19/run_model19.py report

Five fixed arms: original Model 16, raw10, aligned10, raw25, aligned25.
10/25 denotes synthetic training-weight share inside the compact model,
NOT its ensemble weight: the compact ensemble weight always stays at10%.
The original control must reproduce Model 16 OOF margins bit for bit.
Model 18 source checkpoints, model artifacts, source code and input hashes
are verified. Its donors excluded each fitting model's validation fold.
No retraining or new donors are needed to test these existing components.

Freeze the full-ensemble comparison before scoring. Complete real validation
series only, exact TSAUC. Report gains against Model 16 (local0.629300), not
against Model 11 or Model 12. Local and cloud scores are different quantities.
Use 1,000 whole-series paired bootstrap replicates within original folds,
seed190906, shared across all arms. Four-comparison simultaneous one-sided
95% lower bounds from centered bootstrap maximum. Fixed screening gate:
gain >=.0003, wins >=4/5, lower bound >0. Tie-break: lower synthetic share,
then raw before aligned. Fixed sigmoid-margin false alarms at .5 and .8
are diagnostics only; outputs are not calibrated event probabilities.

This is exploratory research on repeatedly reused development folds.
Model 16's upstream selection/refitting was not fully nested. Model 18
already informed this follow-up. The bootstrap is conditional on saved
predictions and cannot account for all adaptive research/training uncertainty.
No reserved outer holdout reopening, competition labels, uploads, full-data
retraining or submission changes. CPU only; resumable checksummed outputs
are isolated under models/model19/runs/.
