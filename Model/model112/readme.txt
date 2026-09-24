Model112: deployable top-level blend of the Submission28 chain and the
oracle-distilled Model111 LightGBM voter  ->  Submission30.

Why
Inside Model84's research stack the distilled voter is diluted (Model111 final
gain ~0). Added at the top level of the *deployable* chain it helps on both
pilot folds (+0.0028 at w=0.5: 0.63236 -> 0.63516), with essentially no
engineering risk: the serving row and incremental whitening row already
computed by Submission28 contain every feature the distilled voters need.

Definition
chain = Submission28 score (Model26 stack with Model62 LightGBM voters, rolling-
        max-128 premium, 0.9/0.1 compact blend) == models/model20/report.py
        ensemble_margin(Model62 voter).
dist  = logit(float32 logit-bag of the five Model111 fold voters) at serving;
        fold-specific OOF voter for development scoring.
score = (chain - mu_chain)/sd_chain + w * (dist - mu_dist)/sd_dist
Constants = pooled development OOF means/stds; w chosen on pilot folds 0-1
from {0.2,...,1.0}; folds 2-4 are confirmation (select_blend.py).

Files
deployable_probe.py  pilot comparison of chains (m73, m84 voter, m111 voter)
select_blend.py      weight selection + constants.json
deployment/          solution112.py, main.py, build.py, validate.py
Build: .venv\Scripts\python.exe models\model112\deployment\build.py
Validate: ...\validate.py parity ; ...\validate.py benchmark --series 24 --workers 2

No upload or submission without the user's emailed approval.
