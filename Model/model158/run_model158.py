"""Model158: soft targets for the SYNTHETIC rows (splices + re-cuts) from the oracle teacher.

In every student so far only the real rows get 0.5*y + 0.5*teacher; splices and re-cuts keep hard labels.
Model151 tried soft re-cut targets with the OLD teacher and failed (that teacher scored only 0.60 on
re-cuts).  The Model157 oracle features are a model-based posterior of the break position, so the teacher
may transfer to glued / re-cut series.  Teacher for held dev fold k: Model152's 4-fold oracle teacher
full_teacherB{k}.txt (trained on every dev fold except k -> never saw fold k), inputs = fold-0 student
columns + 25 Model109 features + 7 oracle features of the (glued / re-cut) series.
  feats           teacher inputs for splice rows (plan k) and re-cut rows -> data/
  student FOLD..  arm SRBS40 = SRB40 (Model157) + soft synthetic targets; prints teacher TS-AUC on the
                  synthetic rows first (continue only if it is not poor)
Usage: python models/model158/run_model158.py feats | student FOLD [FOLD ...]
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
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, RUNS = HERE / "data", HERE / "runs"
M157 = ROOT / "models/model157"
F152 = ROOT / "models/model152/runs"


def feats():
    from models.model109.teacher_features import series_features as tf109
    from models.model157.oracle_bayes import series_oracle
    from models.model20.run_model20 import synthetic_tasks
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    DATA.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        out = DATA / f"splice_teacher_fold{k}.npy"
        if out.exists():
            continue
        tasks, sy, _ = synthetic_tasks(pool, protocol["plans"][k], layout, True)
        rows = []
        for h, o, keep in tasks:
            h = np.asarray(h, np.float64); o = np.asarray(o, np.float64); keep = np.asarray(keep)
            rows.append(np.hstack([tf109(h, o)[keep], series_oracle(h, o)[keep]]))
        X = np.concatenate(rows); assert len(X) == len(sy)
        np.save(out, X); print(f"splice teacher inputs fold {k}: {X.shape}", flush=True)
    out = DATA / "recut_oracle.npy"
    if not out.exists():
        rec = np.load(ROOT / "models/model147/data/recuts.npz")
        parent, shift, t_on = rec["parent"], rec["shift"], rec["time_online"]
        X = np.empty((len(parent), 7), np.float32)
        st = np.flatnonzero(np.r_[True, parent[1:] != parent[:-1]]); en = np.r_[st[1:], len(parent)]
        for a, b in zip(st, en):
            h, o = np.asarray(pool[int(parent[a])][0], np.float64), np.asarray(pool[int(parent[a])][1], np.float64)
            s = int(shift[a]); X[a:b] = series_oracle(np.r_[h, o[:s]], o[s:])[t_on[a:b]]
        np.save(out, X); print("re-cut oracle inputs", X.shape, flush=True)


def student(protocol, layout, real, pool, rec, fold, arm="SRBS40"):
    out = RUNS / f"{arm}_fold{fold}.npz"
    if out.exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    col0 = np.asarray([union.index(n) for n in protocol["contract"]["names"][0]])     # the full teachers' columns
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    ok = (rec["fold"] != fold) & (rec["fold"] != (fold + 1) % 5)
    teacher = lgb.Booster(model_file=str(F152 / f"full_teacherB{fold}.txt"))
    # splices: teacher on [fold-0 columns of the splice rows, 25 + 7 teacher inputs]
    from models.model155.run_model155 import HERE as H155
    su = np.load(H155 / f"data/splice_union_fold{fold}.npy")                            # all 500 union cols of the splices
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    sp = teacher.predict(np.hstack([su[:, col0], np.load(DATA / f"splice_teacher_fold{fold}.npy")])); del su
    rt = np.hstack([rec["X_union"][ok][:, col0], np.load(ROOT / "models/model147/data/recuts_teacher.npy")[ok],
                    np.load(DATA / "recut_oracle.npy")[ok]])
    rp = teacher.predict(rt); del rt
    ry, rtm = rec["target"][ok].astype(np.int8), rec["time_online"][ok]
    print(f"{arm} fold {fold}: teacher TS-AUC on splices {tsauc(sy, sp, st):.4f}, on re-cuts {tsauc(ry, rp, rtm):.4f}", flush=True)
    ex, ey, et = np.concatenate((sx, rec["X_union"][ok][:, columns])), np.r_[sy, ry].astype(np.int8), np.r_[st, rtm].astype(t.dtype)
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    saved = np.load(M157 / f"runs/fold{fold}_teacherB.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * saved["prob"], 0.5 * ey.astype(np.float32) + 0.5 * np.r_[sp, rp].astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{fold}.npy"), rec["split"][ok]])
    keep = np.load(m147.D144 / f"runs_white/E_fold{fold}.npz")["columns"]
    x = np.concatenate((x, extra), axis=1)[:, keep]; del extra
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), np.float32)
    for lo in range(0, len(held), 100_000):
        r = held[lo:lo + 100_000]
        pred[lo:lo + len(r)] = model.predict(np.concatenate((real[r][:, columns], np.asarray(split_real[r])), axis=1)[:, keep])
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held, columns=keep)
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    ref = tsauc(layout.target[held], np.load(M157 / f"runs/SRB40_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs SRB40 {ref:.6f} ({s - ref:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    if sys.argv[1] == "feats":
        feats(); return
    RUNS.mkdir(parents=True, exist_ok=True)
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    rec = dict(np.load(m147.HERE / "data/recuts.npz")); rec["split"] = np.load(m147.HERE / "data/recuts_split.npy")
    for f in [int(a) for a in sys.argv[2:]] or [0]:
        student(protocol, layout, real, pool, rec, f)


if __name__ == "__main__":
    main()
