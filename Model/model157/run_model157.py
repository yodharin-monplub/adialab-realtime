"""Model157 step 2: stronger teacher -> better soft targets for the SRE40 student.

teachers FOLD : Model111's cross-fitting exactly (for held fold k, teacher j not in {k, k+1} is trained on the
                folds outside {k, k+1, j} and scores the fit rows of fold j), same settings/rows/seeds, but the
                privileged inputs are the old 25 Model109 features + the 7 Bayesian oracle features.
                -> runs/fold{k}_teacherB.npz (rows, prob) and a report vs the old teacher.
student FOLD  : Model147 SRE40 unchanged except the real rows' soft target 0.5*y + 0.5*teacherB (arm SRB40).
Usage: python models/model157/run_model157.py teachers|student FOLD [FOLD ...]
"""
from __future__ import annotations

import json
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
from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
M111 = ROOT / "models/model111/runs"
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))
TVAR = __import__("os").environ.get("TVAR", "B")   # teacher variant: B = Model111 settings; C = 127 leaves + 900k rows (teacher_probe.py)
TROWS = 900_000 if TVAR == "C" else 450_000
ALPHA = float(__import__("os").environ.get("ALPHA", "0.5"))   # label share of the soft target (0.5 = Model111)
TPARAMS = dict(m147.m111.TEACHER_PARAMS, **({"num_leaves": 127, "min_child_samples": 200} if TVAR == "C" else {}))


def teachers(protocol, layout, real, old, new, fold):
    out = RUNS / f"fold{fold}_teacher{TVAR}.npz"
    if out.exists():
        return
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, _ = fitting_rows(layout, fold, seed)
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in protocol["contract"]["names"][fold]])
    prob = np.full(len(fit), np.nan, np.float32); report = {}
    oldrep = json.loads((M111 / f"fold{fold}_teacher.json").read_text())
    for j in range(5):
        if j in (fold, (fold + 1) % 5):
            continue
        t0 = time.monotonic()
        pool_rows = np.flatnonzero(~np.isin(layout.row_fold, [fold, (fold + 1) % 5, j]))
        rows = np.sort(np.random.default_rng(7000 + 10 * fold + j).choice(pool_rows, min(TROWS, len(pool_rows)), replace=False))
        x = np.hstack([real[rows][:, columns], np.asarray(old[rows]), np.asarray(new[rows])])
        y, t = layout.target[rows], layout.time_online[rows]
        model = lgb.LGBMClassifier(objective="binary", random_state=31 + j, n_jobs=THREADS, verbosity=-1,
                                   force_col_wise=True, **TPARAMS)
        model.fit(x, y, sample_weight=time_pair_weights(y, t)); del x
        mask = layout.row_fold[fit] == j; tr = fit[mask]
        prob[mask] = model.predict_proba(np.hstack([real[tr][:, columns], np.asarray(old[tr]), np.asarray(new[tr])]))[:, 1]
        report[j] = tsauc(layout.target[tr], prob[mask], layout.time_online[tr])
        print(f"fold {fold} teacher{TVAR} {j}: {report[j]:.5f} (old {oldrep[str(j)]:.5f}) [{time.monotonic() - t0:.0f}s]", flush=True)
    assert np.isfinite(prob).all()
    np.savez(out, rows=fit, prob=prob); (RUNS / f"fold{fold}_teacher{TVAR}.json").write_text(json.dumps(report, indent=2))


def student(protocol, layout, real, pool, rec, fold, arm=None):
    arm = arm or (f"SR{TVAR}40" + ("" if ALPHA == 0.5 else f"a{int(round(ALPHA * 100))}"))
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
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    ex, ey, et = np.concatenate((sx, rec["X_union"][ok][:, columns])), np.r_[sy, rec["target"][ok]].astype(np.int8), \
        np.r_[st, rec["time_online"][ok]].astype(t.dtype)
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    saved = np.load(RUNS / f"fold{fold}_teacher{TVAR}.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[ALPHA * y.astype(np.float32) + (1 - ALPHA) * saved["prob"], ey.astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{fold}.npy"), rec["split"][ok]])
    keep = np.load(m147.D144 / f"runs_white/E_fold{fold}.npz")["columns"]
    x = np.concatenate((x, extra), axis=1)[:, keep]; del extra
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS, verbosity=-1,
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
    ref = tsauc(layout.target[held], np.load(m147.RUNS / f"SRE40_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs SRE40 {ref:.6f} ({s - ref:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    what = sys.argv[1]; folds = [int(a) for a in sys.argv[2:]] or [0]
    RUNS.mkdir(parents=True, exist_ok=True)
    _, protocol, layout, real, pool, old = m147.m111.load_everything()
    if what == "teachers":
        new = np.load(HERE / "data/real.npy", mmap_mode="r")
        if TVAR == "D":   # Model160: + 6 oracle-v2 features (raw-level posterior, lag-2 dependence)
            new = np.hstack([np.asarray(new), np.load(ROOT / "models/model160/data/real.npy")])
        for f in folds:
            teachers(protocol, layout, real, old, new, f)
    else:
        rec = dict(np.load(m147.HERE / "data/recuts.npz")); rec["split"] = np.load(m147.HERE / "data/recuts_split.npy")
        for f in folds:
            student(protocol, layout, real, pool, rec, f)


if __name__ == "__main__":
    main()


