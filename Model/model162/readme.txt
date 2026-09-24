Model162 - the 2025 external series again, now with oracle-teacher soft
targets (2026-09-23).

Model145 found the 481k external rows (2025 challenge, 9,911 series cut into
the real-time format) HURT as hard-labelled rows (fold 1 -0.009 / -0.016;
the student scored only 0.56 on them - a different generator).
ext_teacher.py replays Model145's exact cuts (verified equal: series, step,
tau) and builds the teacher inputs (25 Model109 + 7 oracle features).
The mean of the five 4-fold oracle teachers scores 0.717 TS-AUC on the
external rows (none of them saw any external series).
holdout_ext.py (in memory): the FB1 + soft-synthetic recipe with the
external rows pooled into the non-real block (40% mass), soft targets.
  sealed holdout: student 0.63491 vs 0.63512 (-0.0002),
                  final 0.63585 vs 0.63591 (-0.0001)
-> soft targets remove the harm, but the external series add nothing.
CLOSED. Not used in any bundle.
