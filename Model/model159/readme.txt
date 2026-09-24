Model159 - Submission37 candidate: the Submission36 recipe + soft teacher
targets for the synthetic rows (Model158), 2026-09-22.

prep_soft.py builds the missing teacher inputs:
  dev re-cuts _s148/_s149: 7 oracle features -> models/model158/data/
  holdout re-cuts (3 sets): 25 Model109 + 7 oracle -> models/model154/data/
Voters: models/model154/run_all.py arms AS1/AS2/AS3 (= AB1-AB3 + soft
synthetic targets: splices of plan v by the 4-fold oracle teacher T_v, a dev
re-cut of a fold-j parent by T_j, holdout re-cuts by the dev-only oracle
teacher hold_teacherB). Trained on all 10,000 labelled series.
Evidence (sealed holdout, see Model157/158 readmes):
  Sub36 recipe (BF3) +0.0018 vs Submission33; soft synthetic targets add
  about +0.0004 on top (student alone +0.0024) -> expected ~+0.0022.
Bundle: models/model154/deployment/build35.py with ARMS=AS1,AS2,AS3
DEST=submission37, then validate35.py / seal35.py.
  python models/model159/prep_soft.py
  python models/model154/run_all.py AS1   (AS2, AS3)
