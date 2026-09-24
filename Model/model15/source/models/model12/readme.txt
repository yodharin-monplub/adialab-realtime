Model 12
ExtraTrees diversity model using Model 11's 154 strictly causal features.

See README.md for the detailed model description and dependencies.
Run from the project root: python3 models/run_next_models.py run --models 12
CPU only. Five fixed complete-series development folds. Outer holdout and test labels are excluded. Results and fitted fold models are saved in runs/. Blends with earlier OOF models are exploratory and are not leaderboard scores.
