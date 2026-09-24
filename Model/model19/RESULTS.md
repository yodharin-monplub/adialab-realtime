# Model 19: continue from Model 16

The approximately 0.622 cloud submission is the control. Values below are LOCAL development TSAUC, not cloud scores.

Keep its 90% backbone and persistence unchanged; replace only its 10% compact component with splice-trained fits.

| Replacement | Full ensemble TSAUC | Gain vs Model 16 | Fold wins | Simultaneous 95% lower gain |
|---|---:|---:|---:|---:|
| control | 0.629300 | +0.000000 | 0/5 | — |
| raw_10 | 0.629667 | +0.000367 | 4/5 | -0.000462 |
| aligned_10 | 0.629607 | +0.000307 | 3/5 | -0.000522 |
| raw_25 | 0.629708 | +0.000408 | 4/5 | -0.000421 |
| aligned_25 | 0.629814 | +0.000514 | 4/5 | -0.000316 |

Pass requires gain >= 0.0003, at least 4/5 fold wins, and simultaneous lower bound > 0.
Selected: none; keep original Model 16.

Exploratory continuation on reused development folds, not fresh confirmation. Model 18 augmentation and Model 16 upstream selection are already development-informed. Conditional paired bootstrap excludes upstream training/selection uncertainty.

Full fold gains and fixed-threshold false-alarm counts are saved in the run summary.

Model 16 unchanged; nothing submitted
