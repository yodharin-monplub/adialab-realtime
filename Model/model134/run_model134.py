"""Model134: distilled student trained with block-bootstrap replicas (data/boot.npz).

Arms (pilot folds 0-1, everything else Model111):
  replace : bootstrap rows instead of the splice rows, 40% synthetic cell mass
  add     : splices 40% + bootstrap rows taking 25% of the remaining real mass
Bootstrap rows for held fold k come only from donors in folds outside {k, k+1};
they carry hard labels (like the splices).  Usage: python models/model134/run_model134.py 0 1
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
RUNS = HERE / "runs"
BOOT_ROWS = 400_000


def run_fold(protocol, layout, real, pool, boot, fold, arm):
    out = RUNS / f"{arm}_fold{fold}.npz"
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
    # bootstrap rows for this fold (fold-clean by donor fold), mapped to the fold's 317 columns
    bnames = list(boot["names"])
    bcol = np.asarray([bnames.index(n) for n in names])
    ok = np.flatnonzero(~np.isin(boot["donor_fold"], [fold, (fold + 1) % 5]))
    rng = np.random.default_rng(1340 + fold)
    pick = np.sort(rng.choice(ok, min(BOOT_ROWS, len(ok)), replace=False))
    bx = np.asarray(boot["X"][pick][:, bcol], np.float32); by = boot["target"][pick].astype(np.int8); bt = boot["time_online"][pick]
    saved = np.load(m111.RUNS / f"fold{fold}_teacher.npz")
    assert np.array_equal(saved["rows"], fit)
    target = 0.5 * y.astype(np.float32) + 0.5 * saved["prob"]
    if arm == "replace":
        rw, bw, acc = splice_weights(weights, y, t, by, bt, m111.m84.CONFIG["synthetic_mass"])
        x = np.concatenate((real[fit][:, columns], bx)); labels = np.r_[target, by.astype(np.float32)]; sw_all = np.r_[rw, bw]
    else:
        tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
        rw, sw, _ = splice_weights(weights, y, t, sy, st, m111.m84.CONFIG["synthetic_mass"])
        rw2, bw, acc = splice_weights(rw, y, t, by, bt, 0.25)
        parent20 = json.loads((m111.m84.PARENT20 / "protocol.json").read_text())
        sbase = read_cache(m111.m84.PARENT20, f"aligned_25_fold{fold}_features", {
            "protocol_sha256": m111.m84.CONFIG["model20_protocol_sha256"], "rows": len(sy),
            "names": parent20["contract"]["names"][fold], "tasks": len(tasks)})
        swhite = m111.m84.white_source.cached_white(m111.m84.PARENT57 / f"aligned_25_fold{fold}_white.npy",
            m111.m84.CONFIG["model57_protocol_sha256"], len(sy))
        sx = np.concatenate((np.asarray(sbase), np.asarray(swhite)), axis=1)
        x = np.concatenate((real[fit][:, columns], sx, bx)); labels = np.r_[target, sy.astype(np.float32), by.astype(np.float32)]
        sw_all = np.r_[rw2, sw, bw]
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=8, verbosity=-1,
                              deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])
    t0 = time.monotonic()
    model.fit(x, labels, sample_weight=sw_all)
    del x
    pred = model.predict(real[held][:, columns]).astype(np.float32)
    model.booster_.save_model(str(RUNS / f"{arm}_fold{fold}.txt"))
    np.savez(out, prediction=pred, held_rows=held)
    base = tsauc(layout.target[held], np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"], layout.time_online[held])
    score = tsauc(layout.target[held], pred, layout.time_online[held])
    print(f"fold {fold} {arm}: Model134 raw {score:.6f} vs Model111 {base:.6f} ({score - base:+.6f}); "
          f"bootstrap rows {len(pick):,}, share {acc['actual_share']:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1]
    _, protocol, layout, real, pool, _ = m111.load_everything()
    boot = np.load(HERE / "data" / "boot.npz")
    boot = {k: boot[k] for k in boot.files}
    for arm in ("replace", "add"):
        for fold in folds:
            run_fold(protocol, layout, real, pool, boot, fold, arm)


if __name__ == "__main__":
    main()
