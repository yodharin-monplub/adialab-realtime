"""Model129 pilot (deliberately leaky, decision-only): GRU logit as a tree feature.

Trains the Model111 student for fold 0 with ONE extra column: the OOF logit of
the Model124 GRU ensemble.  Real training rows of fold j use the fold-j GRU
(trained without fold j but WITH fold 0), synthetic rows get NaN.  This is the
standard OOF-stacking leak plus a train/serve mismatch, so it can only answer
"is there a large signal here?"; a real build needs nested GRUs and synthetic
sequences.  Usage: python models/model129/pilot_leaky.py [fold]
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
TAGS = ("", "h96l3_", "h128_")


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main(fold):
    _, protocol, layout, real, pool, teacher = m111.load_everything()
    gru = np.zeros(layout.row_count, dtype=np.float32)
    for k in range(5):
        m = layout.row_fold == k
        gru[m] = np.mean([lg(np.load(ROOT / f"models/model124/remote_runs/{t}fold{k}_pred.npz")["prediction"]) for t in TAGS], axis=0)
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
    sx = np.concatenate((np.asarray(sbase), np.asarray(swhite), np.full((len(sy), 1), np.nan, np.float32)), axis=1)
    saved = np.load(m111.RUNS / f"fold{fold}_teacher.npz")
    assert np.array_equal(saved["rows"], fit)
    target = 0.5 * y.astype(np.float32) + 0.5 * saved["prob"]
    x = np.concatenate((np.hstack([real[fit][:, columns], gru[fit][:, None]]), sx))
    results = {}
    for arm in ("with_gru", "without_gru"):
        cols = slice(None) if arm == "with_gru" else slice(0, 317)
        model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1,
                                  deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])
        t0 = time.monotonic()
        model.fit(x[:, cols], np.r_[target, sy.astype(np.float32)], sample_weight=np.r_[rw, sw])
        xh = np.hstack([real[held][:, columns], gru[held][:, None]])[:, cols]
        pred = model.predict(xh).astype(np.float32)
        results[arm] = tsauc(layout.target[held], pred, layout.time_online[held])
        np.save(HERE / f"pilot_fold{fold}_{arm}.npy", pred)
        print(f"fold {fold} {arm}: raw {results[arm]:.6f} ({time.monotonic() - t0:.0f}s)", flush=True)
        if arm == "with_gru":
            imp = model.booster_.feature_importance("gain"); print(f"  gru column gain share {imp[-1] / imp.sum():.3f}")
    print(f"fold {fold}: gain from GRU column {results['with_gru'] - results['without_gru']:+.6f} "
          f"(GRU alone {tsauc(layout.target[held], gru[held], layout.time_online[held]):.6f})", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
