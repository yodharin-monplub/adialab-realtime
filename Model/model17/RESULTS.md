# Model 17: fixed diversity blend screen

8,333 complete development series / 4,195,720 online rows.

| Model 16 / Model 12 / Model 6 weights | Mean fold TSAUC | Gain vs Model 16 | Fold wins |
| --- | ---: | ---: | ---: |
| 100% / 0% / 0% | 0.629300070 | +0.000000000 | 0/5 |
| 95% / 5% / 0% | 0.629338359 | +0.000038289 | 3/5 |
| 90% / 10% / 0% | 0.629298410 | -0.000001660 | 3/5 |
| 95% / 0% / 5% | 0.629487129 | +0.000187059 | 4/5 |
| 90% / 0% / 10% | 0.629650091 | +0.000350021 | 4/5 |
| 85% / 0% / 15% | 0.629790217 | +0.000490147 | 4/5 |
| 90% / 5% / 5% | 0.629513197 | +0.000213127 | 3/5 |
| 85% / 10% / 5% | 0.629466355 | +0.000166286 | 3/5 |

Best diagnostic: m16_85_m6_15.
Paired whole-series bootstrap 95% interval for its gain: [+0.000049465, +0.000877630] (300 replicates).
Leave-one-fold-out recipe-selection sensitivity: +0.000490147 mean gain; 4/5 wins.

Predeclared gates: gain at least 0.00025, at least 4/5 fold wins, positive bootstrap lower endpoint, and positive leave-fold selection gain.

Candidate clears the development screen; exact deployment-matched independent validation is still required.

Reused development OOF only; leave-fold selection is a sensitivity check, not fully nested validation. Bootstrap conditions on saved predictions and omits adaptive selection uncertainty.

No holdout/test labels were loaded. No models were retrained, deployed, or submitted.

## Independent-validation status

The reserved outer holdout is not genuinely fresh: Model 9's README explicitly
records that it was previously revealed, and Model 7 prohibits using the revealed
Model 6 holdout. It was not reopened. Therefore the positive development screen
is not independent confirmation, and Model 16 remains the deployed incumbent.
See `VALIDATION_STATUS.md`. Six Model 17 and nine shared regression tests passed.
