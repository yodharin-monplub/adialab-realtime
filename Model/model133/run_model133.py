"""Model133: distilled student + 23 causal moment/transform columns (user suggestion).

Model111 recipe (frozen Model84 fit rows, 40% synthetic mass, 317 columns,
cached teacher targets) plus data/real.npy (layout order) and data/syn_fold{k}.npy
(kept synthetic rows).  Usage: python models/model133/run_model133.py 0 1
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
from models.model133.moment_features import NAMES  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def run_fold(protocol, layout, real, pool, fold, extra_real):
    out = RUNS / f"mom_fold{fold}.npz"
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
    extra_syn = np.load(HERE / "data" / f"syn_fold{fold}.npy")
    assert extra_syn.shape == (len(sy), len(NAMES))
    sx = np.concatenate((np.asarray(sbase), np.asarray(swhite), extra_syn), axis=1)
    saved = np.load(m111.RUNS / f"fold{fold}_teacher.npz")
    assert np.array_equal(saved["rows"], fit)
    target = 0.5 * y.astype(np.float32) + 0.5 * saved["prob"]
    x = np.concatenate((np.hstack([real[fit][:, columns], extra_real[fit]]), sx))
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])
    t0 = time.monotonic()
    model.fit(x, np.r_[target, sy.astype(np.float32)], sample_weight=np.r_[rw, sw])
    del x
    pred = model.predict(np.hstack([real[held][:, columns], extra_real[held]])).astype(np.float32)
    model.booster_.save_model(str(RUNS / f"mom_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held)
    base = tsauc(layout.target[held], np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"], layout.time_online[held])
    score = tsauc(layout.target[held], pred, layout.time_online[held])
    imp = model.booster_.feature_importance("gain"); share = imp[-len(NAMES):].sum() / imp.sum()
    top = sorted(zip(imp[-len(NAMES):], NAMES), reverse=True)[:5]
    print(f"fold {fold}: Model133 raw {score:.6f} vs Model111 {base:.6f} ({score - base:+.6f}); new-column gain share "
          f"{share:.3f}; top {[(n, round(float(g) / imp.sum(), 3)) for g, n in top]} ({time.monotonic() - t0:.0f}s)", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    extra_real = np.load(HERE / "data" / "real.npy", mmap_mode="r")
    assert extra_real.shape == (layout.row_count, len(NAMES))
    for fold in folds:
        run_fold(protocol, layout, real, pool, fold, extra_real)


if __name__ == "__main__":
    main()
