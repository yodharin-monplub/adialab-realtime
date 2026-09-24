"""Model156: sealed-holdout check of the evidence student, offline (Model152 components).

  build           3 evidence columns for every holdout row -> models/model156/data/holdout.npy
  score TAG ARM,ARM [RUNSDIR] [CONST]
                  student = logit-bag of the ARMs' fold voters, whose inputs are the voter's pruned column list
                  (names from the Model84 protocol + split tests, as in Model147) followed by the 3 evidence
                  columns; blend exactly as offline_holdout.py; writes models/model152/holdout_student_{TAG}.npz
                  so bootstrap152.py can compare it with Submission33.
Usage: python models/model156/offline_be.py build | score BE40 BE40 [models/model156/runs] [constants_m147.json]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "models/model152"))


def build():
    import pandas as pd
    from models.model156.bayes_evidence import series_features
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    starts = np.flatnonzero(ref["step"] == 0); ends = np.r_[starts[1:], len(ref["step"])]
    X = np.empty((len(ref["step"]), 3), np.float32)
    for sid, a, b in zip(ref["ids"], starts, ends):
        s = frame.loc[int(sid)]
        X[a:b] = series_features(s.loc[s.period == 1, "value"].to_numpy(np.float64), s.loc[s.period == 2, "value"].to_numpy(np.float64))
    np.save(HERE / "data/holdout.npy", X)
    from tsauc import tsauc
    print("saved holdout evidence", X.shape, "single-column TS-AUC", [round(tsauc(ref["target"], X[:, j], ref["step"]), 4) for j in range(3)])


def score(tag, arms, runs, const_file):
    import lightgbm as lgb
    import offline_holdout as oh
    from split_tests import NAMES
    from tsauc import tsauc
    protocol = json.loads(oh.PROTOCOL.read_text())
    index = oh.x_columns()
    X = np.load(oh.H / "X.npy", mmap_mode="r"); ev = np.load(HERE / "data/holdout.npy")
    n = len(ev); z = np.zeros(n); count = 0
    for arm in arms:
        for k in range(5):
            names = list(protocol["contract"]["names"][k]) + list(NAMES)
            cols = np.load(runs / f"{arm}_fold{k}.npz")["columns"]
            idx = np.asarray([index[names[c]] for c in cols])
            booster = lgb.Booster(model_file=str(runs / f"{arm}_fold{k}.txt")); assert booster.num_feature() == len(idx) + 3
            p = np.empty(n)
            for lo in range(0, n, 200_000):
                p[lo:lo + 200_000] = booster.predict(np.concatenate((np.asarray(X[lo:lo + 200_000][:, idx]), ev[lo:lo + 200_000]), axis=1), num_threads=8)
            p = np.clip(p.astype(np.float32).astype(np.float64), 1e-7, 1 - 1e-7)
            z += np.log(p) - np.log1p(-p); count += 1
    z /= count
    pe = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6); dist = np.log(pe) - np.log1p(-pe)
    bc = oh.bundle_constants(); const = json.loads((ROOT / "models/model112" / const_file).read_text())
    chain, gru = np.load(oh.H / "chain.npy"), np.load(oh.H / "gru.npy")
    s = ((chain - bc["mu_chain"]) / bc["sd_chain"] + bc["weight"] * (dist - const["mu_dist"]) / const["sd_dist"]
         + bc["weight_gru"] * (gru - bc["mu_gru"]) / bc["sd_gru"])
    final = 1 / (1 + np.exp(-s))
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz"); y, t = ref["target"], ref["step"]
    bag3 = np.load(ROOT / "models/model147/holdout_predictions_BAG3.npz")["final"]
    res = dict(tag=tag, student_raw=tsauc(y, dist, t), final=tsauc(y, final, t), bag3=tsauc(y, bag3, t))
    res["gain_vs_bag3"] = res["final"] - res["bag3"]
    np.savez(ROOT / f"models/model152/holdout_student_{tag}.npz", dist=dist.astype(np.float32), final=final.astype(np.float32), rows=np.arange(n))
    (HERE / f"holdout_{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build":
        build()
    else:
        score(a[1], a[2].split(","), ROOT / (a[3] if len(a) > 3 else "models/model156/runs"), a[4] if len(a) > 4 else "constants_m147.json")
