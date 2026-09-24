Model88: width-5 whitened early-break specialist.

The frozen Model57 whitening ladder starts at width 16. An independent
competition-specific report measured a width-5 feature as materially stronger
than a large detector bank at online ages 1--5, while recent online CUSUM
research likewise motivates statistics weighted for early detection. Model88
tests that narrow missing scale rather than adding another broad detector bank.

Eight causal columns are appended to Model84's 317-column contract: the seven
existing Rosenblatt-whitened innovation channels accumulated over the last
five observations, plus log2(online age + 1) so the tree can route the short
signal by elapsed time. Whitening is fit only on each series' history. The
real/synthetic rows, 40% synthetic TSAUC cell mass, held-series exclusions,
LightGBM expressive recipe, and causal postprocessing remain frozen.

Pilot folds 0--1 select a fixed 0.25, 0.50, or 0.75 blend of the new processed
signal with Model84, the current 0.637888 leader. A +0.0001 pilot gain is
required before folds 2--4 train. Final promotion requires full and
confirmation TSAUC >= 0.640, confirmation gain >= 0.0005, 2/3 confirmation
wins, positive paired-series bootstrap lower 95%, and positive fixed-ID-half
gains. CPU only, resumable caches and fold checkpoints. No cloud activity.

Research sources:
  https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206
  https://doi.org/10.1016/j.jeconom.2025.106071

Run from repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/model88-mpl .venv/bin/python -u models/model88/run_model88.py
