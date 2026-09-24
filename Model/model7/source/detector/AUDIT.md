# Detector audit

Generated: `2026-08-26T19:25:22+07:00` by `python3 detector/audit_detectors.py`.

Overall result: **PASS**.

## Correctness fixes

- Restored `detector/core/channels18.py`, the missing history-normalized d3 innovation helper required by d19 and d20.
- Replaced d16's invalid positional d4 prefix slice with an explicit feature-name mapping. d4 inserts `x99_up`/`x99_dn` at column 90, so the old slice fed the wrong channels to d16.
- Versioned only d16's specialist implementation. Core, candidate, and novel production signatures remain unchanged.
- Exact-constant history exposes floating-point cancellation differences between some old vectorized and recurrent accumulators. Outputs remain finite, deterministic, and causal. This edge case is documented rather than changing accepted production semantics; none of the 8,000 training histories is near-constant.

## Verification summary

- Detector sources discovered: **49** (28 core, 21 candidate).
- Compile/import issues: **0**.
- Local module self-tests passed: **47**; central-only modules: **2**.
- Project regression tests: **66 passed**.
- Composite synthetic cases: **35/35 passed**; instance isolation **5/5 passed**.
- Minimum train-history standard deviation: **0.99999998**; near-constant histories: **0**.

## Detector inventory

| Detector | Folder | Role | Features | Direct detector dependencies | d3 prefix | Self-test |
|---|---|---|---:|---|---|---|
| d1 | core | foundation/ablation | 31 | none | no | pass |
| d2 | core | foundation/ablation | 154 | none | no | pass |
| d3 | core | foundation/ablation | 236 | none | yes | pass |
| d4 | core | production core | 306 | none | no | pass |
| d5 | core | foundation/ablation | 254 | detectors3 | yes | pass |
| d6 | core | foundation/ablation | 258 | detectors3 | yes | pass |
| d7 | core | foundation/ablation | 285 | detectors3, detectors6 | yes | pass |
| d8 | core | foundation/ablation | 296 | detectors3, detectors6 | yes | pass |
| d9 | core | foundation/ablation | 489 | detectors3 | yes | pass |
| d10 | core | specialist | 284 | detectors3 | yes | pass |
| d11 | core | specialist | 396 | detectors3 | yes | pass |
| d12 | core | specialist | 80 | none | no | pass |
| d13 | core | foundation/ablation | 558 | detectors3 | yes | pass |
| d15 | core | production core | 558 | detectors3 | yes | covered_by_central_suite |
| d16 | core | specialist | 416 | detectors3 | yes | pass |
| d17 | core | foundation/ablation | 558 | detectors3, detectors15 | yes | pass |
| d19 | core | foundation/ablation | 30 in audit probe; dynamic (n_templates) | detectors3, channels18 | no | pass |
| d20 | core | foundation/ablation | 24 | detectors3, detectors15, channels18, detectors19 | no | pass |
| d21 | core | specialist | 54 | none | no | pass |
| d22 | core | production core | 230 | detectors3, detectors15 | no | pass |
| d23 | core | specialist | 310 | detectors3 | no | pass |
| d24 | core | specialist | 60 | detectors3 | no | pass |
| d25 | core | specialist | 200 | none | no | pass |
| d26 | core | production core | 184 | detectors3, detectors15 | no | pass |
| d27 | core | production core | 70 | detectors4 | no | pass |
| d28 | core | specialist | 120 | detectors3 | no | pass |
| d29 | core | specialist | 92 | detectors3, detectors15 | no | pass |
| d30 | core | production core | 56 | detectors4, detectors15 | no | pass |
| d31 | candidate | foundation/ablation | 30 in audit probe; dynamic (n_templates) | none | no | pass |
| d32 | candidate | production candidate | 110 | none | no | pass |
| d33 | candidate | production candidate | 112 | none | no | pass |
| d35 | candidate | production candidate | 128 | none | no | pass |
| d36 | candidate | specialist | 230 | detectors3, detectors15 | no | pass |
| d37 | candidate | production candidate | 126 | detectors3 | no | pass |
| d40 | candidate | production candidate | 324 | none | no | pass |
| d45 | candidate | production candidate | 230 | detectors3, detectors15 | no | pass |
| d51 | candidate | production candidate | 30 | detectors3, detectors15 | no | pass |
| d52 | candidate | production candidate | 64 | detectors3, detectors15 | no | pass |
| d53 | candidate | production candidate | 14 | detectors3 | no | pass |
| d55 | candidate | production candidate | 132 | detectors3, detectors15 | no | pass |
| d57 | candidate | production candidate | 110 | detectors3 | no | pass |
| d58 | candidate | production candidate | 110 | detectors3, detectors57 | no | pass |
| d60 | candidate | novel/rejected | 129 | none | no | covered_by_central_suite |
| d61 | candidate | foundation/ablation | 28 | none | no | pass |
| d62 | candidate | foundation/ablation | 36 | none | no | pass |
| d63 | candidate | foundation/ablation | 30 | none | no | pass |
| d64 | candidate | foundation/ablation | 49 | none | no | pass |
| d65 | candidate | foundation/ablation | 36 | detectors61, detectors64 | no | pass |
| d66 | candidate | foundation/ablation | 60 | none | no | pass |

## Repeated detector logic

A detector file can intentionally emit many features from one core innovation. Prefix, upward/downward CUSUM, fixed windows, EWMAs, and absolute-value summaries are different time geometries, not accidental copies. The following standalone experiments prepend the complete d3 feature block: `d3, d5, d6, d7, d8, d9, d10, d11, d13, d15, d16, d17`.

The active composite contract avoids that repeated base block: d10, d11, d15, and d16 are sliced to retain only their additions, while d3's shared statistics come from d4. Feature names are bank-prefixed and no active contract contains duplicate names.

## Reproduce

```bash
python3 detector/audit_detectors.py --cache-scan full
python3 tests/run_tests.py
```

The audit checks implementation correctness, causality, determinism, and redundancy. It does not claim that every retained experimental detector improves TSAUC; model selection remains a separate leakage-free validation task.
