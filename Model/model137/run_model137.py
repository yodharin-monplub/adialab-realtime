"""Model137 (stage 1: DATA). Same model (Model111 distilled student), different training data.

Arms (all five folds, raw held-fold TS-AUC):
  orig     : sampled original rows only (400k), no synthetic
  m20/m60  : original + splice synthetic at 20% / 60% cell mass (Model130 pilot files reused)
  allrows  : ALL original rows of the fitting folds (~2.5M, no 20% sampling) + splices 40%
  m40      : current (Model111) - not retrained, its OOF files are the reference
Usage: python models/model137/run_model137.py [arm ...]   (default: all arms)
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model115.run_model115 import teacher_for_rows  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks  # noqa: E402
from models.model21.run_model21 import read_cache  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def student(seed):
    return lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1,
                             deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])


def synthetic_block(protocol, layout, pool, fold, columns_names):
    plan = protocol["plans"][fold]
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    parent20 = json.loads((m111.m84.PARENT20 / "protocol.json").read_text())
    sbase = read_cache(m111.m84.PARENT20, f"aligned_25_fold{fold}_features", {
        "protocol_sha256": m111.m84.CONFIG["model20_protocol_sha256"], "rows": len(sy),
        "names": parent20["contract"]["names"][fold], "tasks": len(tasks)})
    swhite = m111.m84.white_source.cached_white(m111.m84.PARENT57 / f"aligned_25_fold{fold}_white.npy",
        m111.m84.CONFIG["model57_protocol_sha256"], len(sy))
    return np.concatenate((np.asarray(sbase), np.asarray(swhite)), axis=1), sy, st


def finish(model, layout, real, columns, fold, arm, started):
    held = np.flatnonzero(layout.row_fold == fold)
    pred = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 100_000):
        rows = held[left:left + 100_000]
        pred[left:left + len(rows)] = model.predict(real[rows][:, columns])
    np.savez(RUNS / f"{arm}_fold{fold}.npz", prediction=pred, held_rows=held)
    ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
    s, b = tsauc(layout.target[held], pred, layout.time_online[held]), tsauc(layout.target[held], ref, layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs m40 (Model111) {b:.6f} ({s - b:+.6f}) [{time.monotonic() - started:.0f}s]", flush=True)


def run_orig(protocol, layout, real, fold):
    if (RUNS / f"orig_fold{fold}.npz").exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    saved = np.load(m111.RUNS / f"fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
    target = 0.5 * y.astype(np.float32) + 0.5 * saved["prob"]
    model = student(seed); model.fit(real[fit][:, columns], target, sample_weight=w)
    finish(model, layout, real, columns, fold, "orig", t0)


def run_allrows(protocol, layout, real, pool, teacher, fold):
    if (RUNS / f"allrows_fold{fold}.npz").exists():
        return
    t0 = time.monotonic()
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fitting = (layout.row_fold != fold) & (layout.row_fold != (fold + 1) % 5)
    fit = np.flatnonzero(fitting)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, y, t)
    sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
    rw, sw, _ = splice_weights(weights, y, t, sy, st, 0.4)
    m111.THREADS = 8
    soft = teacher_for_rows(layout, real, teacher, columns, fold, fit)
    target = 0.5 * y.astype(np.float32) + 0.5 * soft
    x = np.empty((len(fit) + len(sy), len(columns)), dtype=np.float32)
    for left in range(0, len(fit), 200_000):
        r = fit[left:left + 200_000]; x[left:left + len(r)] = real[r][:, columns]
    x[len(fit):] = sx; del sx
    print(f"allrows fold {fold}: {len(fit):,} real rows + {len(sy):,} synthetic", flush=True)
    model = student(seed); model.fit(x, np.r_[target, sy.astype(np.float32)], sample_weight=np.r_[rw, sw]); del x
    finish(model, layout, real, columns, fold, "allrows", t0)


def run_mass(protocol, layout, real, pool, teacher, fold, mass):
    arm = f"m{int(mass * 100):02d}"
    if (RUNS / f"{arm}_fold{fold}.npz").exists():
        return
    old = ROOT / f"models/model130/runs/a050{arm}_fold{fold}.npz"
    if old.exists():
        shutil.copy2(old, RUNS / f"{arm}_fold{fold}.npz")
        d = np.load(old); held = d["held_rows"]
        ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
        s = tsauc(layout.target[held], d["prediction"], layout.time_online[held]); b = tsauc(layout.target[held], ref, layout.time_online[held])
        print(f"{arm} fold {fold}: {s:.6f} vs m40 {b:.6f} ({s - b:+.6f}) [reused Model130]", flush=True)
        return
    t0 = time.monotonic()
    m111.m84.CONFIG["synthetic_mass"] = mass; m111.RUNS = RUNS; m111.TAG = f"tmp{arm}"; m111.THREADS = 8
    for f in range(5):
        src = ROOT / f"models/model111/runs/fold{f}_teacher.npz"
        if src.exists() and not (RUNS / src.name).exists():
            shutil.copy2(src, RUNS / src.name)
    pred = m111.run_fold(protocol, layout, real, pool, teacher, fold)
    (RUNS / f"tmp{arm}_fold{fold}.npz").rename(RUNS / f"{arm}_fold{fold}.npz")
    held = np.flatnonzero(layout.row_fold == fold)
    ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
    s, b = tsauc(layout.target[held], pred, layout.time_online[held]), tsauc(layout.target[held], ref, layout.time_online[held])
    print(f"{arm} fold {fold}: {s:.6f} vs m40 {b:.6f} ({s - b:+.6f}) [{time.monotonic() - t0:.0f}s]", flush=True)
    m111.m84.CONFIG["synthetic_mass"] = 0.4


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or ["orig", "m20", "m60", "allrows"]
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    teacher_runs = m111.RUNS
    for arm in arms:
        for fold in range(5):
            m111.RUNS = teacher_runs
            if arm == "orig":
                run_orig(protocol, layout, real, fold)
            elif arm == "allrows":
                run_allrows(protocol, layout, real, pool, teacher, fold)
            elif arm in ("m20", "m60"):
                run_mass(protocol, layout, real, pool, teacher, fold, int(arm[1:]) / 100)


if __name__ == "__main__":
    main()
