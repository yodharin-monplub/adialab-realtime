Model30 — lower-shift matched augmentation without context columns

Model27's matched synthetic backbone helped fold 0 but failed badly on fold 1,
and a pilot-trained reliability gate did not recover fold 2. The likely failure
mode is fold-specific reliance on the 13 clean-history context columns.

Model30 reuses exactly Model27's frozen positive and hard-negative composites,
but removes those context columns. It also lowers synthetic mass from 25% to
15%, restores depth 3, and fixes a 50/50 probability blend with Model26. This
tests the augmentation signal itself with substantially less distribution shift.

All five predictions are held-series OOF. Promotion requires mean TSAUC >= .640,
gain >= .0007, at least 4/5 fold wins, positive whole-series bootstrap lower
bound, and positive gain in both fixed ID-hash halves.

CPU only. No test labels or automatic cloud submission.
