"""Model152: full-development refit of the SRE40 student (all 8,333 dev series instead of 3/5 per voter).

Learning curve (run_model152.py, LC2): 3 -> 2 fitting folds costs -0.0039 +/- 0.0016 (4/5), so the
student is still series-limited.  The deployed student averages 5 fold voters that each saw 60% of the
series; here every voter sees ALL dev series (the sealed holdout stays out), and the result is judged on
the sealed holdout offline (offline_holdout.py) against the same recipe's fold voters (SRE40 / BAG3).

teachers  : 5 new cross-fitted teachers T_j (Model111 settings, 450k rows from the 4 dev folds != j, fold-0
            student columns + 25 Model109 features); T_j gives the soft target of every sampled row of fold j.
voter v   : Model147 SRE40 recipe with fold v's column set and seed, trained on 400k*5/3 sampled rows of
            ALL dev folds (soft targets 0.5*y + 0.5*T), fold v's 1,000 splices (only those have cached
            features for that column set) + all re-cuts of the chosen set, 40% non-real mass.
Arms: FULL (re-cut set ""), FULLb (_s148), FULLc (_s149) -> runs/{arm}_fold{v}.txt/.npz (npz: columns).
Usage: python models/model152/run_full.py teachers | ARM [voters]
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
from models.model20.run_model20 import CONFIG, reference_weights, sampled_mask, splice_weights  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SETS = {"FULL": "", "FULLb": "_s148", "FULLc": "_s149", "FULLB": "", "FULLBb": "_s148", "FULLBc": "_s149",
        "FB1": ""}   # FB1 = oracle-teacher set "" (FULLB collides with FULLb on case-insensitive Windows file names)
ORACLE = __import__("os").environ.get("ORACLE", "0") == "1"   # Model157: teachers also see the 7 Bayesian oracle features
TEACHER_FILE = "full_teacherB.npz" if ORACLE else "full_teacher.npz"
ROWS = CONFIG["fit_rows"] * 5 // 3
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))


def all_dev_rows(layout, seed):
    dev = layout.row_fold >= 0
    eligible = np.flatnonzero(dev & sampled_mask(layout.series_ids[layout.series_position], layout.time_online))
    return eligible if len(eligible) <= ROWS else np.sort(np.random.default_rng(seed + 1).choice(eligible, ROWS, replace=False))


def teachers(protocol, layout, real, teacher):
    out = RUNS / TEACHER_FILE
    if out.exists():
        return
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in protocol["contract"]["names"][0]])
    target_rows = np.flatnonzero(layout.row_fold >= 0)            # every dev row gets a cross-fitted soft target
    prob = np.full(len(target_rows), np.nan, np.float32)
    for j in range(5):
        t0 = time.monotonic()
        pool_rows = np.flatnonzero((layout.row_fold >= 0) & (layout.row_fold != j))
        rows = np.sort(np.random.default_rng(9000 + j).choice(pool_rows, 450_000, replace=False))
        x = np.hstack([real[rows][:, columns], np.asarray(teacher[rows])])
        y, t = layout.target[rows], layout.time_online[rows]
        model = lgb.LGBMClassifier(objective="binary", random_state=31 + j, n_jobs=THREADS, verbosity=-1,
                                   force_col_wise=True, **m147.m111.TEACHER_PARAMS)
        model.fit(x, y, sample_weight=time_pair_weights(y, t)); del x
        mask = layout.row_fold[target_rows] == j
        for lo in range(0, int(mask.sum()), 200_000):
            r = target_rows[mask][lo:lo + 200_000]
            idx = np.flatnonzero(mask)[lo:lo + 200_000]
            prob[idx] = model.predict_proba(np.hstack([real[r][:, columns], np.asarray(teacher[r])]))[:, 1]
        rj = target_rows[mask]
        print(f"full teacher {j}: {tsauc(layout.target[rj], prob[mask], layout.time_online[rj]):.5f} on fold {j} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        model.booster_.save_model(str(RUNS / f"{TEACHER_FILE[:-4]}{j}.txt"))
    assert np.isfinite(prob).all()
    np.savez(out, rows=target_rows, prob=prob)


def voter(protocol, layout, real, pool, rec, v, arm):
    out = RUNS / f"{arm}_fold{v}.npz"
    if out.exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][v]["seed"]
    fit = all_dev_rows(layout, seed)
    names, union = protocol["contract"]["names"][v], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    dev = np.flatnonzero(layout.row_fold >= 0)
    w = reference_weights(y, t, layout.target[dev], layout.time_online[dev])
    saved = np.load(RUNS / ("full_teacherB.npz" if arm.startswith(("FULLB", "FB")) else "full_teacher.npz"))
    soft = saved["prob"][np.searchsorted(saved["rows"], fit)]
    ex, ey, et = rec["X_union"][:, columns], rec["target"].astype(np.int8), rec["time_online"].astype(t.dtype)
    sx, sy, st = synthetic_block(protocol, layout, pool, v, names)
    ex, ey, et = np.concatenate((sx, ex)), np.r_[sy, ey], np.r_[st, et]
    rw, sw, _ = splice_weights(w, y, t, ey, et, 0.4)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * soft, ey.astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    split_real = np.load(m147.D144 / "data_white/real.npy", mmap_mode="r")
    extra = np.concatenate([np.asarray(split_real[fit]), np.load(m147.D144 / f"data_white/syn_fold{v}.npy"), rec["split"]])
    assert len(extra) == len(x) == len(target)
    keep_cols = np.load(m147.D144 / f"runs_white/E_fold{v}.npz")["columns"]
    x = np.concatenate((x, extra), axis=1)[:, keep_cols]; del extra
    print(f"{arm} voter {v}: {len(fit):,} real rows, {len(sy):,} splice rows, {len(rec['target']):,} re-cut rows", flush=True)
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
    if ORACLE:
        teacher = np.hstack([np.asarray(teacher), np.load(ROOT / "models/model157/data/real.npy")])
    if args[0] == "teachers":
        teachers(protocol, layout, real, teacher); return
    arm = args[0]
    tag = SETS[arm]
    rec = dict(np.load(m147.HERE / f"data/recuts{tag}.npz")); rec["split"] = np.load(m147.HERE / f"data/recuts{tag}_split.npy")
    for v in ([int(a) for a in args[1:]] or range(5)):
        voter(protocol, layout, real, pool, rec, v, arm)


if __name__ == "__main__":
    main()


