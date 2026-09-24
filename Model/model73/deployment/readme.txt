Submission27: exploratory Model73 direct LightGBM voter.

This inference-only package replaces the Model26 core feature voter with the
five frozen 40%-synthetic LightGBM voters from Model62. The remaining Model26
stack is unchanged. It computes the same 61 causal whitening features at each
online prefix and never reads training labels in the cloud.

Local reused-development five-fold OOF: 0.6354292998 versus 0.6324069786
for Model26. This is not Model62's full research score, and the cloud score
may differ. The outer holdout and competition test labels are sealed.

forceFirstTrain=false; trainFrequency=0; 12 inference workers, one numerical
thread per worker. CPU only. Benchmark-derived runtime and RAM evidence are
recorded separately before upload.
