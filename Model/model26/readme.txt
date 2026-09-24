Model26 — tree-depth diversity for the cloud-confirmed Model23
============================================================
Runner and reporter implemented. Training first reproduces the depth3
baseline and stops on any mismatch before fitting a depth4 candidate.
Run/resume: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python models/model26/run_model26.py run
Current progress is recorded in training.log and runs/latest.json.

Reference: Model23 cloud TS-AUC 0.6237894019876368, run114854,
prediction93349. Its full development CV is 0.6320947258732259.

Hypothesis: depth4 trees may learn interactions between break statistics
that depth3 misses. Mixing depths may retain the robustness of Model23.
This is a model-capacity experiment, not another synthetic-weight search.

Predeclared single candidate:
- Train depth4 versions of BOTH aligned25 and aligned60, five folds each.
- Keep the original 400,000 real plus 400,000 synthetic rows per fit.
- Reuse the exact Model20 donor pairs, cached features, seeds and row order.
- Keep 800 rounds and all remaining hyperparameters unchanged.
- Within each fold, first average the two depth4 probabilities equally.
- Average that result equally with Model23's paired depth3 probability.
- Pass the resulting component through the unchanged stack and postprocess.
- No depth, weight, threshold, or round-count selection after seeing scores.

Validation requirements before training:
1. Verify upstream protocols and read-only cache block checksums.
2. Freeze this specification, runner, reporter and tests in a hashed protocol.
3. Reproduce one depth3 aligned25 fold bit-for-bit with the new runner.
4. Test donor exclusion and held-row alignment, including stopping fold.
5. Test exact float32 mixture ordering and checkpoint resume integrity.

Promotion: FULL ensemble gain >=0.0003 versus Model23, at least four of
five fold wins, and positive paired whole-series bootstrap 95% lower bound
using 1,000 draws (seed260911). One prespecified candidate/contrast.
Component score and no-break alarms are diagnostic, not selection criteria.
All validation uses real series only. Reserved outer holdout stays closed.
Development folds have been reused; conditional bootstrap does not account
for upstream selection or training uncertainty. A local pass is not proof
of a cloud gain. Retain Model23 unless cloud evidence supports replacement.

CPU only, four training threads, sequential fits. Estimate runtime from the
first fit and record peak memory. While training, wait and check every ten
minutes; notify only completion, failure or required action. Do not change
any earlier model or submission. No automatic cloud upload in this study.
