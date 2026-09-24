# Model 2: Train-from-Scratch Ensemble

Model 2 retained the two-family LightGBM architecture but trained both models
inside Crunch rather than uploading pretrained weights. Its reusable training
implementation is the repository-root `ensemble_training.py` together with the
shared feature and validation modules.

The frozen historical bundle remains in
[`submissions/submission_2/`](../../submissions/submission_2/). A complete
source-only copy is stored in [`source/`](source/), including the exact
manifested detector and train-from-scratch implementation. The retained
development driver under `source/research/` is not claimed as a frozen
historical revision. Generated models, caches, predictions, and training data
are deliberately excluded.
