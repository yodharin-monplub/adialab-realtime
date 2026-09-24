"""Model151 runner: distilled re-cuts.  Identical to Model147 SRE40 except that re-cut rows get the soft
target 0.5*y + 0.5*teacher instead of the hard label.

Teacher for held fold k and a re-cut whose parent is in fold j: Model111's saved fold{k}_teacher{j}.txt
(trained on folds other than k, k+1, j) on [the fold's 317 student columns, 25 Model109 teacher features
of the re-cut series] - the same cross-fitting as the real rows' soft targets.  Splices keep hard labels.
Arms: SRD40 / SRD40b / SRD40c = re-cut sets "" / _s148 / _s149 (compare with SRE40 / b / c).
Usage: python models/model151/run_model151.py ARM [ARM ...] [folds]
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model147.run_model147 as m147  # noqa: E402  (installs Linux stubs via Model111)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
M111 = ROOT / "models/model111/runs"
SETS = {"SRD40": "", "SRD40b": "_s148", "SRD40c": "_s149",
        "SRDS40": "", "SRDS40b": "_s148", "SRDS40c": "_s149"}   # SRDS: splices get soft targets too
ALPHA = 0.5


def splice_teacher(protocol, layout, pool, columns, fold, sx_student):
    """Soft targets for the splices: mean of the fold's three saved teachers on [student cols, teacher feats of
    the glued series].  Donors are in fitting folds only, so no teacher has seen the held fold."""
    from models.model20.run_model20 import synthetic_tasks
    from models.model109.teacher_features import series_features
    tasks, sy, _ = synthetic_tasks(pool, protocol["plans"][fold], layout, True)
    tf = np.concatenate([series_features(np.asarray(h, np.float64), np.asarray(o, np.float64))[keep] for h, o, keep in tasks])
    assert len(tf) == len(sx_student) == len(sy)
    x = np.hstack([sx_student, tf])
    probs = [lgb.Booster(model_file=str(p)).predict(x) for p in sorted(M111.glob(f"fold{fold}_teacher[0-4].txt"))]
    assert len(probs) == 3
    return np.mean(probs, axis=0).astype(np.float32)


def recut_teacher(rec, tfeat, ok, columns, fold):
    """Cross-fitted teacher probability for every usable re-cut row of held fold `fold`."""
    prob = np.full(int(ok.sum()), np.nan, np.float32)
    folds = rec["fold"][ok]
    x = np.hstack([rec["X_union"][ok][:, columns], tfeat[ok]])
    for j in np.unique(folds):
        booster = lgb.Booster(model_file=str(M111 / f"fold{fold}_teacher{int(j)}.txt"))
        assert booster.num_feature() == x.shape[1], (booster.num_feature(), x.shape)
        m = folds == j
        prob[m] = booster.predict(x[m])
    assert np.isfinite(prob).all()
    return prob


def run_fold(protocol, layout, real, pool, rec, tfeat, fold, arm):
    out = RUNS / f"{arm}_fold{fold}.npz"
    if out.exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    ok = (rec["fold"] != fold) & (rec["fold"] != (fold + 1) % 5)
    tprob = recut_teacher(rec, tfeat, ok, columns, fold)
    ry = rec["target"][ok].astype(np.int8)
    print(f"{arm} fold {fold}: re-cut teacher TS-AUC {tsauc(ry, tprob, rec['time_online'][ok]):.4f} on {len(ry):,} rows", flush=True)
    ex, ey, et = rec["X_union"][ok][:, columns], ry, rec["time_online"][ok].astype(t.dtype)
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    s_target = sy.astype(np.float32)
    if arm.startswith("SRDS"):
        sprob = splice_teacher(protocol, layout, pool, columns, fold, sx)
        print(f"{arm} fold {fold}: splice teacher TS-AUC {tsauc(sy, sprob, st):.4f} on {len(sy):,} rows", flush=True)
        s_target = ALPHA * s_target + (1 - ALPHA) * sprob
    ex, ey, et = np.concatenate((sx, ex)), np.r_[sy, ey], np.r_[st, et]
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    saved = np.load(M111 / f"fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[ALPHA * y.astype(np.float32) + (1 - ALPHA) * saved["prob"],
                   s_target, ALPHA * ry.astype(np.float32) + (1 - ALPHA) * tprob]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{fold}.npy"), rec["split"][ok]])
    assert len(extra) == len(x) == len(target), (extra.shape, x.shape, target.shape)
    keep_cols = np.load(m147.D144 / f"runs_white/E_fold{fold}.npz")["columns"]
    x = np.concatenate((x, extra), axis=1)[:, keep_cols]; del extra
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=m147.THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m147.m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        xh = np.concatenate((real[rows][:, columns], np.asarray(split_real[rows])), axis=1)[:, keep_cols]
        pred[left:left + len(rows)] = model.predict(xh)
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held, columns=keep_cols)
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    b = tsauc(layout.target[held], np.load(M111 / f"a050_fold{fold}.npz")["prediction"], layout.time_online[held])
    ref = np.load(m147.RUNS / f"SRE40{arm.split('40', 1)[1]}_fold{fold}.npz")["prediction"]
    r = tsauc(layout.target[held], ref, layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs current (Model111) {b:.6f} ({s - b:+.6f}); vs hard-label twin {r:.6f} "
          f"({s - r:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    args = sys.argv[1:]
    RUNS.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args if not a.isdigit()] or ["SRD40"]
    folds = [int(a) for a in args if a.isdigit()] or [0, 1]
    _, protocol, layout, real, pool, _ = m147.m111.load_everything()
    for arm in arms:
        tag = SETS[arm]
        rec = dict(np.load(m147.HERE / f"data/recuts{tag}.npz"))
        rec["split"] = np.load(m147.HERE / f"data/recuts{tag}_split.npy")
        tfeat = np.load(m147.HERE / f"data/recuts{tag}_teacher.npy")
        for fold in folds:
            run_fold(protocol, layout, real, pool, rec, tfeat, fold, arm)


if __name__ == "__main__":
    main()
