"""Model154 step 2: ALL-data student voters for the final bundle = Model152 FULL recipe + the 1,667 holdout
series (their sampled real rows, their re-cuts), i.e. all 10,000 labelled series.

The recipe was chosen and verified on the sealed holdout (Model152 BAGF3 +0.0013 vs Submission33); this
step only adds 20% more series, so it can no longer be scored on the holdout (its predictions there are
in-sample).  Soft targets: dev rows from Model152's 4-fold teachers (full_teacher.npz); holdout rows from a
teacher trained on dev rows only (all 5 folds, same Model111 settings) -> still out-of-sample for them.
Arms ALL / ALLb / ALLc (re-cut sets "", _s148, _s149 for dev and the matching holdout sets).
Usage: python models/model154/run_all.py teacher | ARM [voters]
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model147.run_model147 as m147  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model152.run_full import all_dev_rows  # noqa: E402
from models.model20.run_model20 import reference_weights, splice_weights  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, RUNS = HERE / "data", HERE / "runs"
F152 = ROOT / "models/model152/runs"
SETS = {"ALL": "", "ALLb": "_s148", "ALLc": "_s149", "AB1": "", "AB2": "_s148", "AB3": "_s149",
        "AS1": "", "AS2": "_s148", "AS3": "_s149",   # AS* = AB* + soft teacher targets for splices and re-cuts (Model158/159)
        "AP1": ["", "_s148", "_s149"], "AP2": ["", "_s148", "_s149"], "AP3": ["", "_s148", "_s149"]}   # AP2/AP3 = AP1 with SYNA=0.3; AP3 also fixes the oracle-teacher flag for AP arms   # AP1 = AS recipe with the three re-cut sets POOLED in every voter (Model163)
# AB* = Model157 oracle teachers (dev rows: Model152 full_teacherB.npz; holdout rows: hold_teacherB.npz)
ORACLE = __import__("os").environ.get("ORACLE", "0") == "1"
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))


def holdout_teacher(protocol, layout, real, teacher):
    out = RUNS / ("hold_teacherB.npz" if ORACLE else "hold_teacher.npz")
    if out.exists():
        return
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in protocol["contract"]["names"][0]])
    rows = np.sort(np.random.default_rng(9100).choice(np.flatnonzero(layout.row_fold >= 0), 450_000, replace=False))
    extra_dev = np.load(ROOT / "models/model157/data/real.npy", mmap_mode="r") if ORACLE else None
    x = np.hstack([real[rows][:, columns], np.asarray(teacher[rows])] + ([np.asarray(extra_dev[rows])] if ORACLE else []))
    y, t = layout.target[rows], layout.time_online[rows]
    t0 = time.monotonic()
    model = lgb.LGBMClassifier(objective="binary", random_state=36, n_jobs=THREADS, verbosity=-1,
                               force_col_wise=True, **m147.m111.TEACHER_PARAMS)
    model.fit(x, y, sample_weight=time_pair_weights(y, t)); del x
    h = np.load(DATA / "hold_rows.npz")
    hx = [h["X_union"][:, columns], h["teacher_feat"]] + ([np.load(ROOT / "models/model157/data/holdout.npy")[h["rows"]]] if ORACLE else [])
    prob = model.predict_proba(np.hstack(hx))[:, 1].astype(np.float32)
    print(f"holdout teacher: TS-AUC {tsauc(h['target'], prob, h['time_online']):.5f} on {len(prob):,} holdout rows "
          f"(out-of-sample) [{time.monotonic() - t0:.0f}s]", flush=True)
    model.booster_.save_model(str(RUNS / ("hold_teacherB.txt" if ORACLE else "hold_teacher.txt")))
    np.savez(out, prob=prob)


def load_sets(tag):
    """One re-cut set (tag str) or several pooled (list of tags): dev re-cuts, holdout re-cuts, teacher inputs."""
    tags = [tag] if isinstance(tag, str) else list(tag)
    dev = [dict(np.load(m147.HERE / f"data/recuts{t}.npz")) for t in tags]
    rec = {k: np.concatenate([d[k] for d in dev]) for k in ("X_union", "target", "time_online", "fold")}
    rec["split"] = np.concatenate([np.load(m147.HERE / f"data/recuts{t}_split.npy") for t in tags])
    rec["teacher"] = np.concatenate([np.hstack([np.load(ROOT / f"models/model147/data/recuts{t}_teacher.npy"),
                                                np.load(ROOT / f"models/model158/data/recut_oracle{t}.npy")]) for t in tags])
    hold = [dict(np.load(DATA / f"recuts_hold{t}.npz")) for t in tags]
    hrec = {k: np.concatenate([d[k] for d in hold]) for k in ("X_union", "split", "target", "time_online")}
    hrec["teacher"] = np.concatenate([np.load(DATA / f"recuts_hold{t}_teacher.npy") for t in tags])
    return rec, hrec


def synthetic_soft(protocol, rec, hrec, v, tag, n_splice):
    """Teacher probabilities for [splices of plan v, dev re-cuts, holdout re-cuts] (Model158 recipe).
    Splices: 4-fold oracle teacher T_v (never saw fold v; plan v's donors avoid folds v, v+1).  Dev re-cut of a
    fold-j parent: T_j (never saw fold j).  Holdout re-cuts: the dev-only oracle teacher hold_teacherB."""
    union = protocol["contract"]["union_names"]
    col0 = np.asarray([union.index(n) for n in protocol["contract"]["names"][0]])
    T = {j: lgb.Booster(model_file=str(F152 / f"full_teacherB{j}.txt")) for j in range(5)}
    su = np.load(ROOT / f"models/model155/data/splice_union_fold{v}.npy")
    sp = T[v].predict(np.hstack([su[:, col0], np.load(ROOT / f"models/model158/data/splice_teacher_fold{v}.npy")]))
    assert len(sp) == n_splice
    rt = np.hstack([rec["X_union"][:, col0], rec["teacher"]])
    rp = np.empty(len(rt))
    for j in range(5):
        m = rec["fold"] == j
        rp[m] = T[j].predict(rt[m])
    rt = None
    hp = lgb.Booster(model_file=str(RUNS / "hold_teacherB.txt")).predict(
        np.hstack([hrec["X_union"][:, col0], hrec["teacher"]]))
    return np.r_[sp, rp, hp].astype(np.float32)


def voter(protocol, layout, real, pool, rec, hrec, v, arm):
    out = RUNS / f"{arm}_fold{v}.npz"
    if out.exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][v]["seed"]
    names, union = protocol["contract"]["names"][v], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    fit = all_dev_rows(layout, seed)
    oracle = arm.startswith(("AB", "AS", "AP"))   # AP* (pooled re-cut sets) also use the Model157 oracle teachers
    h = np.load(DATA / "hold_rows.npz"); hsoft = np.load(RUNS / ("hold_teacherB.npz" if oracle else "hold_teacher.npz"))["prob"]
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    y = np.r_[layout.target[fit], h["target"]].astype(np.int8)
    t = np.r_[layout.time_online[fit], h["time_online"]].astype(layout.time_online.dtype)
    dev = np.flatnonzero(layout.row_fold >= 0)
    w = reference_weights(y, t, np.r_[layout.target[dev], ref["target"]].astype(np.int8),
                          np.r_[layout.time_online[dev], ref["step"]].astype(layout.time_online.dtype))
    ft = np.load(F152 / ("full_teacherB.npz" if oracle else "full_teacher.npz"))
    soft = np.r_[ft["prob"][np.searchsorted(ft["rows"], fit)], hsoft]
    ex = np.concatenate((rec["X_union"][:, columns], hrec["X_union"][:, columns]))
    ey = np.r_[rec["target"], hrec["target"]].astype(np.int8)
    et = np.r_[rec["time_online"], hrec["time_online"]].astype(t.dtype)
    sx, sy, st = synthetic_block(protocol, layout, pool, v, names)
    ex, ey, et = np.concatenate((sx, ex)), np.r_[sy, ey], np.r_[st, et]
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    syn = ey.astype(np.float32)
    if arm.startswith(("AS", "AP")):
        sa = float(__import__("os").environ.get("SYNA", "0.5"))   # label share of the synthetic rows (Model163: 0.3 slightly better)
        syn = sa * syn + (1 - sa) * synthetic_soft(protocol, rec, hrec, v, SETS[arm], len(sy))
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * soft, syn]
    x = np.concatenate((real[fit][:, columns], h["X_union"][:, columns], ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), h["split"], np.load(m147.D144 / f"data_white/syn_fold{v}.npy"),
                            rec["split"], hrec["split"]])
    assert len(extra) == len(x) == len(target) == len(rw) + len(sw)
    keep_cols = np.load(m147.D144 / f"runs_white/E_fold{v}.npz")["columns"]
    x = np.concatenate((x, extra), axis=1)[:, keep_cols]; del extra
    print(f"{arm} voter {v}: {len(fit):,} dev + {len(h['target']):,} holdout real rows, {len(sy):,} splice rows, "
          f"{len(rec['target']):,} + {len(hrec['target']):,} re-cut rows", flush=True)
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    model.booster_.save_model(str(RUNS / f"{arm}_fold{v}.txt"))
    np.savez(out, columns=keep_cols)
    print(f"{arm} voter {v} done [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    args = sys.argv[1:]
    RUNS.mkdir(parents=True, exist_ok=True)
    _, protocol, layout, real, pool, teacher = m147.m111.load_everything()
    if args[0] == "teacher":
        holdout_teacher(protocol, layout, real, teacher); return
    arm = args[0]
    rec, hrec = load_sets(SETS[arm])
    for v in ([int(a) for a in args[1:]] or range(5)):
        voter(protocol, layout, real, pool, rec, hrec, v, arm)


if __name__ == "__main__":
    main()



