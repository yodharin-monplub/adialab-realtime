Model24: training-size diversity
===============================
Fixed mixtures of Model20 aligned25 and Model22 all_real_plus_synthetic:
mix_25 uses 25% Model22 probability; mix_50 uses 50% Model22 probability.
Both use held-fold predictions before the unchanged ensemble stack.
References: cloud-confirmed Model20 and locally selected Model23 mix_60.
No new fitting, no changes to existing models, no automatic submission.
Freeze source/checkpoint hashes before evaluation. Five real development folds,
1000 paired whole-series bootstrap draws; four simultaneous contrasts (each
candidate against each reference). Promotion requires gain >=0.0003, at least
4/5 fold wins, and positive simultaneous95% lower bound against BOTH references.
This is exploratory reused-CV research after observing Model22's standalone
failure, not independent confirmation. Outer holdout and test labels stay closed.
CPU only. Run: .venv/bin/python models/model24/run_model24.py run
