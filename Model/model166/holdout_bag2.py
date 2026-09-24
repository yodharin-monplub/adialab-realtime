"""Model166: does BAGGING still help now that each student pools three re-cut sets?

Two pooled students (5 voters each) with different re-cut set combinations, trained in memory on the
development series with the Submission39 recipe (oracle-teacher soft targets everywhere, SYNA label share),
then scored on the sealed holdout alone and as a 10-voter logit-mean bag.
  A: "", _s148, _s149      B: _s153, _s154, _s148
Usage: python models/model166/holdout_bag2.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "models/model152"))

import numpy as np  # noqa: E402
import lightgbm as lgb  # noqa: E402

import models.model147.run_model147 as m147  # noqa: E402
from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model152.run_full import all_dev_rows  # noqa: E402
from models.model20.run_model20 import reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

SYNA = float(os.environ.get("SYNA", "0.3"))
SETS = {"A": ["", "_s148", "_s149"], "B": ["_s153", "_s154", "_s148"]}


def pooled(tags):
    rec = {k: np.concatenate([np.load(ROOT / f"models/model147/data/recuts{t}.npz")[k] for t in tags])
           for k in ("X_union", "target", "time_online", "fold")}
    rec["split"] = np.concatenate([np.load(ROOT / f"models/model147/data/recuts{t}_split.npy") for t in tags])
    rec["teacher"] = np.concatenate([np.hstack([np.load(ROOT / f"models/model147/data/recuts{t}_teacher.npy"),
                                                np.load(ROOT / f"models/model158/data/recut_oracle{t}.npy")]) for t in tags])
    return rec


def train_set(protocol, layout, real, pool, rec, T, ft, split_real, col0):
    boosters = {}
    rt = np.hstack([rec["X_union"][:, col0], rec["teacher"]])
    rp = np.empty(len(rt))
    for j in range(5):
        mk = rec["fold"] == j
        rp[mk] = T[j].predict(rt[mk])
    rt = None
    union = protocol["contract"]["union_names"]
    for v in range(5):
        seed = protocol["contract"]["recipes"][v]["seed"]
        names = protocol["contract"]["names"][v]; columns = np.asarray([union.index(n) for n in names])
        fit = all_dev_rows(layout, seed); y, t = layout.target[fit], layout.time_online[fit]
        dev = np.flatnonzero(layout.row_fold >= 0)
        w = reference_weights(y, t, layout.target[dev], layout.time_online[dev])
        soft = ft["prob"][np.searchsorted(ft["rows"], fit)]
        sx, sy, st = synthetic_block(protocol, layout, pool, v, names)
        su = np.load(ROOT / f"models/model155/data/splice_union_fold{v}.npy")
        sp = T[v].predict(np.hstack([su[:, col0], np.load(ROOT / f"models/model158/data/splice_teacher_fold{v}.npy")])); su = None
        ex = np.concatenate((sx, rec["X_union"][:, columns]))
        ey = np.r_[sy, rec["target"]].astype(np.int8)
        et = np.r_[st, rec["time_online"]].astype(t.dtype)
        rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
        target = np.r_[0.5 * y + 0.5 * soft, SYNA * ey + (1 - SYNA) * np.r_[sp, rp]].astype(np.float32)
        x = np.concatenate((real[fit][:, columns], ex)); ex = None
        extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{v}.npy"), rec["split"]])
        keep = np.load(m147.D144 / f"runs_white/E_fold{v}.npz")["columns"]
        x = np.concatenate((x, extra), axis=1)[:, keep]; extra = None
        model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1, deterministic=True,
                                  force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
        model.fit(x, target, sample_weight=np.r_[rw, sw]); x = None
        boosters[v] = model.booster_
        print("  voter", v, "trained", flush=True)
    return boosters


def logits(boosters, protocol, index, X, n):
    from split_tests import NAMES
    z = np.zeros(n)
    for v, b in boosters.items():
        nm = list(protocol["contract"]["names"][v]) + list(NAMES)
        keep = np.load(m147.D144 / f"runs_white/E_fold{v}.npz")["columns"]
        idx = np.asarray([index[nm[c]] for c in keep])
        p = np.empty(n)
        for lo in range(0, n, 200_000):
            p[lo:lo + 200_000] = b.predict(np.asarray(X[lo:lo + 200_000][:, idx]), num_threads=8)
        p = np.clip(p.astype(np.float32).astype(np.float64), 1e-7, 1 - 1e-7)
        z += np.log(p) - np.log1p(-p)
    return z / len(boosters)


def main():
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    union = protocol["contract"]["union_names"]
    col0 = np.asarray([union.index(n) for n in protocol["contract"]["names"][0]])
    T = {j: lgb.Booster(model_file=str(ROOT / f"models/model152/runs/full_teacherB{j}.txt")) for j in range(5)}
    ft = np.load(ROOT / "models/model152/runs/full_teacherB.npz")
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    import offline_holdout as oh
    index = oh.x_columns(); X = np.load(oh.H / "X.npy", mmap_mode="r"); n = X.shape[0]
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz"); y, t = ref["target"], ref["step"]
    bc = oh.bundle_constants(); const = json.loads((ROOT / "models/model112/constants_m147.json").read_text())
    chain, gru = np.load(oh.H / "chain.npy"), np.load(oh.H / "gru.npy")
    zs = {}
    for name, tags in SETS.items():
        print(f"student {name}: sets {tags}", flush=True)
        zs[name] = logits(train_set(protocol, layout, real, pool, pooled(tags), T, ft, split_real, col0),
                          protocol, index, X, n)
    def report(tag, z):
        pe = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6); d = np.log(pe) - np.log1p(-pe)
        out = []
        for ws in (0.5, 1.5):
            f = 1 / (1 + np.exp(-((chain - bc["mu_chain"]) / bc["sd_chain"] + ws * (d - const["mu_dist"]) / const["sd_dist"]
                                  + bc["weight_gru"] * (gru - bc["mu_gru"]) / bc["sd_gru"])))
            out.append(f"ws={ws} {tsauc(y, f, t):.6f}")
        print(f"HOLDOUT {tag}: student raw {tsauc(y, d, t):.6f} | " + " | ".join(out), flush=True)
    report("A (5 voters)", zs["A"]); report("B (5 voters)", zs["B"])
    report("BAG A+B (10 voters)", 0.5 * (zs["A"] + zs["B"]))


if __name__ == "__main__":
    main()
