"""Model147 runner: add "move the start" re-cuts of this year's series to the training rows.

Same student / columns (317) / soft targets / folds / weights as Model111.  Re-cut rows carry hard
labels (like the splices) and are only taken from parents in the FITTING folds of the held fold
(parent fold not in {k, k+1}), so no held series leaks in through its re-cut.
Arms (weight share of the non-real rows, per (step, label) cell as before):
  R40   re-cuts replace the splices                 SR40  splices + re-cuts pooled at 40%
  SR60  splices + re-cuts pooled at 60%
Usage: python models/model147/run_model147.py ARM [ARM ...] [folds]  |  report
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import models.model140.run_model140 as m140  # noqa: E402  (report())
from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))
ARMS = {"R40": (False, 0.4), "SR40": (True, 0.4), "SR60": (True, 0.6),
        "SRE40": (True, 0.4),   # SR40 data + the Model144 whitened split-test columns (that fold's pruned column set)
        "SRE40x2": (True, 0.4), "SRE40x3": (True, 0.4), "SRE50x3": (True, 0.5)}   # 2 / 3 independent re-cut sets
SETS = {"SRE40x2": ["", "_s148"], "SRE40x3": ["", "_s148", "_s149"], "SRE50x3": ["", "_s148", "_s149"],
        "SRE40b": ["_s148"], "SRE40c": ["_s149"]}   # same recipe as SRE40 on a different single re-cut set (for bagging)
ARMS.update({"SRE40b": (True, 0.4), "SRE40c": (True, 0.4)})
SETS["SRE40f"] = ["_f150"]; ARMS["SRE40f"] = (True, 0.4)     # Model150: sign-flipped re-cuts (build_recuts.py --flip --seed 150)
SETS["SRE40g"] = ["_f151"]; ARMS["SRE40g"] = (True, 0.4)     # further flipped sets, seeds 151 / 152
SETS["SRE40h"] = ["_f152"]; ARMS["SRE40h"] = (True, 0.4)
D144 = ROOT / "models/model144"


def run_fold(protocol, layout, real, pool, rec, fold, arm):
    out = RUNS / f"{arm}_fold{fold}.npz"
    if out.exists():
        return
    use_splices, mass = ARMS[arm]
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    ok = (rec["fold"] != fold) & (rec["fold"] != (fold + 1) % 5)
    ex, ey, et = rec["X_union"][ok][:, columns], rec["target"][ok].astype(np.int8), rec["time_online"][ok].astype(t.dtype)
    if use_splices:
        sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
        ex, ey, et = np.concatenate((sx, ex)), np.r_[sy, ey], np.r_[st, et]
    rw, sw, _ = splice_weights(w, y, t, ey, et, mass)
    saved = np.load(ROOT / f"models/model111/runs/fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    target = np.r_[0.5 * y.astype(np.float32) + 0.5 * saved["prob"], ey.astype(np.float32)]
    x = np.concatenate((real[fit][:, columns], ex)); del ex
    keep_cols, held_extra = None, None
    if arm.startswith("SRE"):
        split_real = np.load(D144 / "data_white/real.npy", mmap_mode="r")
        blocks = [np.asarray(split_real[fit]), np.load(D144 / f"data_white/syn_fold{fold}.npy"), rec["split"][ok]]
        extra = np.concatenate(blocks); assert len(extra) == len(x), (extra.shape, x.shape)
        keep_cols = np.load(D144 / f"runs_white/E_fold{fold}.npz")["columns"]
        x = np.concatenate((x, extra), axis=1)[:, keep_cols]; del extra, blocks
        held_extra = split_real
    print(f"{arm} fold {fold}: {len(fit):,} real + {len(ey):,} non-real rows ({int(ok.sum()):,} re-cut), "
          f"non-real weight share {sw.sum() / (rw.sum() + sw.sum()):.3f}", flush=True)
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])
    model.fit(x, target, sample_weight=np.r_[rw, sw]); del x
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        xh = real[rows][:, columns]
        if keep_cols is not None:
            xh = np.concatenate((xh, np.asarray(held_extra[rows])), axis=1)[:, keep_cols]
        pred[left:left + len(rows)] = model.predict(xh)
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held, **({} if keep_cols is None else {"columns": keep_cols}))
    s = tsauc(layout.target[held], pred, layout.time_online[held])
    b = tsauc(layout.target[held], np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"], layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs current (Model111) {b:.6f} ({s - b:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)


def main():
    args = sys.argv[1:]
    m140.RUNS = RUNS; m140.HERE = HERE
    if args == ["report"]:
        m140.report(); return
    RUNS.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args if not a.isdigit()] or ["SR40"]
    folds = [int(a) for a in args if a.isdigit()] or [0, 1]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    cache = {}

    def recuts(tags):
        key = tuple(tags)
        if key not in cache:
            parts = [dict(np.load(HERE / f"data/recuts{tag}.npz")) for tag in tags]
            rec = {k: np.concatenate([p[k] for p in parts]) for k in ("X_union", "target", "time_online", "fold")}
            rec["split"] = np.concatenate([np.load(HERE / f"data/recuts{tag}_split.npy") for tag in tags])
            cache.clear(); cache[key] = rec
        return cache[key]

    for fold in folds:
        for arm in arms:
            run_fold(protocol, layout, real, pool, recuts(SETS.get(arm, [""])), fold, arm)


if __name__ == "__main__":
    main()
