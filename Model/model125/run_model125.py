"""Model125: soft targets for the synthetic splice rows as well.

Model111 distils only the real rows; the 400k aligned synthetic rows (40% of
the training mass) keep hard labels.  Here the fold's saved teachers (Model111
runs, trained on folds outside {k, k+1, j}) also score the synthetic series:
privileged features are computed from each spliced (history, online) pair with
models/model109/teacher_features.py, the three teachers' probabilities are
averaged, and the synthetic targets become 0.5*y + 0.5*teacher too.  Fit
rows, synthetic rows, 40% synthetic cell mass, 317 fold columns, weights and
the expressive LightGBM recipe are Model111's.

Usage: python models/model125/run_model125.py [fold ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model109.teacher_features import NAMES, series_features  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks  # noqa: E402
from models.model21.run_model21 import read_cache  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ALPHA = 0.5
TAG = "a050syn"
THREADS = 8


def synthetic_privileged(tasks, expected_rows):
    """Privileged (future-seeing) features for the kept rows of each synthetic task."""
    out = np.empty((expected_rows, len(NAMES)), dtype=np.float32)
    at = 0
    for history, online, keep in tasks:
        feats = series_features(np.asarray(history, np.float64), np.asarray(online, np.float64))
        out[at:at + len(keep)] = feats[np.asarray(keep)]
        at += len(keep)
    if at != expected_rows or not np.isfinite(out).all():
        raise ValueError("synthetic privileged features misaligned or non-finite")
    return out


def synthetic_soft(fold, sx, spriv, sy, st):
    path = RUNS / f"fold{fold}_synthetic_teacher.npz"
    if path.exists():
        return np.load(path)["prob"]
    x = np.hstack([sx, spriv])
    acc = np.zeros(len(sy), dtype=np.float64)
    used = 0
    for j in range(5):
        model_path = m111.RUNS / f"fold{fold}_teacher{j}.txt"
        if not model_path.exists():
            continue
        booster = lgb.Booster(model_file=str(model_path))
        acc += booster.predict(x, num_threads=THREADS)
        used += 1
    if used != 3:
        raise ValueError(f"expected three saved teachers for fold {fold}, found {used}")
    prob = (acc / used).astype(np.float32)
    print(f"fold {fold}: teacher TS-AUC on synthetic rows {tsauc(sy, prob, st):.5f}", flush=True)
    np.savez(path, prob=prob)
    return prob


def run_fold(protocol, layout, real, pool, teacher, fold):
    out = RUNS / f"{TAG}_fold{fold}.npz"
    if out.exists():
        return np.load(out)["prediction"]
    held = np.flatnonzero(layout.row_fold == fold)
    plan = protocol["plans"][fold]
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    rw, sw, _ = splice_weights(weights, y, t, sy, st, m111.m84.CONFIG["synthetic_mass"])
    parent20 = json.loads((m111.m84.PARENT20 / "protocol.json").read_text())
    sbase = read_cache(m111.m84.PARENT20, f"aligned_25_fold{fold}_features", {
        "protocol_sha256": m111.m84.CONFIG["model20_protocol_sha256"], "rows": len(sy),
        "names": parent20["contract"]["names"][fold], "tasks": len(tasks)})
    swhite = m111.m84.white_source.cached_white(m111.m84.PARENT57 / f"aligned_25_fold{fold}_white.npy",
        m111.m84.CONFIG["model57_protocol_sha256"], len(sy))
    sx = np.concatenate((np.asarray(sbase), np.asarray(swhite)), axis=1)
    assert sx.shape == (len(sy), 317)
    del sbase, swhite
    started = time.monotonic()
    spriv = synthetic_privileged(tasks, len(sy))
    print(f"fold {fold}: synthetic privileged features ({time.monotonic() - started:.0f}s)", flush=True)
    del tasks
    soft_syn = synthetic_soft(fold, sx, spriv, sy, st)
    del spriv
    saved = np.load(m111.RUNS / f"fold{fold}_teacher.npz")
    assert np.array_equal(saved["rows"], fit)
    target_real = ALPHA * y.astype(np.float32) + (1 - ALPHA) * saved["prob"]
    target_syn = ALPHA * sy.astype(np.float32) + (1 - ALPHA) * soft_syn
    x = np.concatenate((real[fit][:, columns], sx))
    del sx
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS,
                              verbosity=-1, deterministic=True, force_col_wise=True,
                              **m111.m84.CONFIG["arms"]["expressive"])
    started = time.monotonic()
    model.fit(x, np.r_[target_real, target_syn], sample_weight=np.r_[rw, sw])
    del x
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        prediction[left:left + len(rows)] = model.predict(real[rows][:, columns])
    model.booster_.save_model(str(RUNS / f"{TAG}_fold{fold}.txt"))
    np.savez(out, prediction=prediction, held_rows=held)
    ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    base = tsauc(layout.target[held], ref, layout.time_online[held])
    print(f"fold {fold}: Model125 raw {score:.6f} vs Model111 {base:.6f} ({score - base:+.6f}, "
          f"{time.monotonic() - started:.0f}s fit)", flush=True)
    return prediction


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    for fold in folds:
        run_fold(protocol, layout, real, pool, teacher, fold)


if __name__ == "__main__":
    main()
