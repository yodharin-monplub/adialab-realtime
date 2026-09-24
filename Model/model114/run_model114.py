"""Model114: oracle-distilled version of Model64's 500-feature LightGBM voter.

Model64 recipe (all 439 detector columns + 61 whitening columns, 31 leaves,
800 trees, 40% aligned synthetic mass, same frozen fit rows and donor plans)
with real-row targets 0.5*y + 0.5*teacher.  Teacher probabilities are the
Model111 cached cross-fitted values for the identical fit rows.

Usage: python models/model114/run_model114.py [fold ...]
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
from models.model57.whitening import FEATURE_NAMES  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
M64 = ROOT / "models/model64/runs/ec1f695c2a8a3a09"
ALPHA = 0.5
TAG = "a050"
THREADS = 6


def run_fold(protocol, layout, real, pool, fold, p64):
    out = RUNS / f"{TAG}_fold{fold}.npz"
    if out.exists():
        return
    held = np.flatnonzero(layout.row_fold == fold)
    plan = protocol["plans"][fold]
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    names = p64["contract"]["names"][fold]
    union = p64["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in names])
    assert len(columns) == 500 and names[-61:] == list(FEATURE_NAMES)
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    rw, sw, _ = splice_weights(weights, y, t, sy, st, 0.4)
    parent20 = json.loads((m111.m84.PARENT20 / "protocol.json").read_text())
    base = read_cache(m111.m84.PARENT20, f"aligned_25_fold{fold}_features", {
        "protocol_sha256": m111.m84.CONFIG["model20_protocol_sha256"], "rows": len(sy),
        "names": parent20["contract"]["names"][fold], "tasks": len(tasks)})
    missing = np.load(M64 / f"aligned_40_fold{fold}_missing.npy", mmap_mode="r")
    white = m111.m84.white_source.cached_white(m111.m84.PARENT57 / f"aligned_25_fold{fold}_white.npy",
        m111.m84.CONFIG["model57_protocol_sha256"], len(sy))
    sx = np.concatenate((np.asarray(base), np.asarray(missing), np.asarray(white)), axis=1)
    assert sx.shape == (len(sy), 500)
    del tasks, base, missing, white
    saved = np.load(m111.RUNS / f"fold{fold}_teacher.npz")
    assert np.array_equal(saved["rows"], fit)
    target = ALPHA * y.astype(np.float32) + (1 - ALPHA) * saved["prob"]
    x = np.concatenate((real[fit][:, columns], sx))
    del sx
    params = json.loads((ROOT / "models/model64/prespec.json").read_text())["lightgbm"]
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS,
                              verbosity=-1, deterministic=True, force_col_wise=True, **params)
    started = time.monotonic()
    model.fit(x, np.r_[target, sy.astype(np.float32)], sample_weight=np.r_[rw, sw])
    del x
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        prediction[left:left + len(rows)] = model.predict(real[rows][:, columns])
    model.booster_.save_model(str(RUNS / f"{TAG}_fold{fold}.txt"))
    np.savez(out, prediction=prediction, held_rows=held)
    reference = np.load(M64 / f"lightgbm_aligned_40_fold{fold}.npz")["prediction"]
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    ref = tsauc(layout.target[held], reference, layout.time_online[held])
    print(f"fold {fold}: Model114 raw {score:.6f} vs Model64 raw {ref:.6f} ({time.monotonic() - started:.0f}s)",
          flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    p64 = json.loads((M64 / "protocol.json").read_text())
    for fold in folds:
        run_fold(protocol, layout, real, pool, fold, p64)


if __name__ == "__main__":
    main()
