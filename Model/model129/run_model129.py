"""Model129 stage 2c: distilled LightGBM student with a nested GRU-logit column.

For held fold k (fold-clean everywhere):
  real training rows of fold j  <- nested GRU (k, j).pred_j           (never saw k or j)
  synthetic rows of fold k plan <- mean_j nested (k, j).pred_syn[keep] (never saw k)
  held fold k rows              <- mean_j nested (k, j).pred_k         (never saw k)
Everything else is Model111 (frozen Model84 recipe, 40% synthetic mass,
317 columns, teacher targets).  Reports raw component vs Model111 and saves
runs/nested_fold{k}.npz.  Usage: python models/model129/run_model129.py [fold ...]
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

from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks  # noqa: E402
from models.model21.run_model21 import read_cache  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
NESTED = HERE / "nested"   # nested_k{k}_j{j}.npz downloaded from the GPU box
RUNS = HERE / "runs"


def nested_features(layout, fold, fit, held, keep_index):
    js = [j for j in range(5) if j not in (fold, (fold + 1) % 5)]
    real_col = np.full(layout.row_count, np.nan, np.float32)
    held_acc = np.zeros(len(held), np.float64)
    syn_acc = None
    for j in js:
        d = np.load(NESTED / f"nested_k{fold}_j{j}.npz")
        real_col[d["rows_j"]] = d["pred_j"]
        assert np.array_equal(d["rows_k"], held)
        held_acc += d["pred_k"]
        syn = d["pred_syn"][keep_index]
        syn_acc = syn.astype(np.float64) if syn_acc is None else syn_acc + syn
    fit_col = real_col[fit]
    assert np.isfinite(fit_col).all(), "nested GRU feature missing on some fit rows"
    return fit_col.astype(np.float32), (held_acc / len(js)).astype(np.float32), (syn_acc / len(js)).astype(np.float32)


def run_fold(protocol, layout, real, pool, fold):
    out = RUNS / f"nested_fold{fold}.npz"
    if out.exists():
        return
    held = np.flatnonzero(layout.row_fold == fold)
    plan = protocol["plans"][fold]
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
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
    keep_index = np.load(HERE / "data" / f"syn_fold{fold}.npz")["keep_index"]
    fit_col, held_col, syn_col = nested_features(layout, fold, fit, held, keep_index)
    sx = np.concatenate((np.asarray(sbase), np.asarray(swhite), syn_col[:, None]), axis=1)
    saved = np.load(m111.RUNS / f"fold{fold}_teacher.npz")
    assert np.array_equal(saved["rows"], fit)
    target = 0.5 * y.astype(np.float32) + 0.5 * saved["prob"]
    x = np.concatenate((np.hstack([real[fit][:, columns], fit_col[:, None]]), sx))
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])
    t0 = time.monotonic()
    model.fit(x, np.r_[target, sy.astype(np.float32)], sample_weight=np.r_[rw, sw])
    del x
    pred = model.predict(np.hstack([real[held][:, columns], held_col[:, None]])).astype(np.float32)
    model.booster_.save_model(str(RUNS / f"nested_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held, gru_feature_held=held_col)
    base = tsauc(layout.target[held], np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"], layout.time_online[held])
    score = tsauc(layout.target[held], pred, layout.time_online[held])
    imp = model.booster_.feature_importance("gain")
    print(f"fold {fold}: Model129 raw {score:.6f} vs Model111 {base:.6f} ({score - base:+.6f}); "
          f"GRU column gain share {imp[-1] / imp.sum():.3f}; nested GRU alone {tsauc(layout.target[held], held_col, layout.time_online[held]):.5f} "
          f"({time.monotonic() - t0:.0f}s)", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for fold in folds:
        run_fold(protocol, layout, real, pool, fold)


if __name__ == "__main__":
    main()
