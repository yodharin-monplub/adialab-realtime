# Model 20: retrain a deployed Model 16 backbone

Core XGBoost, original fold-specific 256 features, depth3, 800 rounds, 400k real fitting rows.
The raw10/aligned25 arms add up to400k synthetic rows; original weights are conserved by (step,target).

| Arm | Component TSAUC | Full ensemble TSAUC | Gain vs incumbent | Gain vs fresh control | Pass |
|---|---:|---:|---:|---:|---|
| Original Model 16 | — | 0.629300 | — | — | incumbent |
| control | 0.611618 | 0.629300 | +0.000000 | +0.000000 | control |
| raw_10 | 0.615214 | 0.630253 | +0.000953 | +0.000953 | True |
| aligned_25 | 0.618173 | 0.631576 | +0.002275 | +0.002275 | True |

Pass requires full-ensemble gain >=0.0003, >=4/5 fold wins, and simultaneous lower bound >0 against BOTH fresh control and original Model16.
Selected: aligned_25.

Reused development experiment, not fresh confirmation. Existing feature rankings and stack weights remain fixed and development-informed. Conditional bootstrap excludes upstream refitting/selection uncertainty. No reserved holdout or test-label evaluation.

These are local development scores, not new cloud scores. Component refits may differ from historical fits due to dependency versions; the fresh paired control isolates augmentation.

Model 16 unchanged; no submission performed by the research runner
