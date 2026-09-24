Model85: add the formerly reserved stopping-fold regimes to Model62 training.

Every earlier Model62-family fit excludes the held fold and the adjacent fold
that once served as a stopping set, so each voter sees only three of five
series folds. Model70 added many rows from those same three folds and failed.
Model85 instead keeps the exact frozen 400,000 real rows and adds all rows from
the adjacent fold passing Model20's original deterministic 20% sampler. Thus
the held fold remains completely unseen while the fit gains a fourth regime
quarter. Reference weights are recomputed from all four non-held folds. The
400,000 aligned synthetic rows, 40% cell mass, 317 features, and standard
Model62 LightGBM recipe are unchanged.

Pilot folds 0-1 select a fixed 0.25, 0.50, or 0.75 blend of the processed new
voter with Model84, the current 0.637888 OOF leader. A +0.0001 pilot gain is
required before folds 2-4 train. Promotion requires confirmation and full OOF
TSAUC >=0.640, confirmation gain >=0.0005, 2/3 wins, positive bootstrap lower
95%, and positive fixed-ID-half gains. CPU only; no cloud or submission.
