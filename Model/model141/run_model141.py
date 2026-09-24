"""Model141 - STAGE 5 (model): hyperparameter random search and seed bags.

Everything except the LightGBM parameters is Model111's (rows, targets, 40%
splices, 317 columns).  Search on pilot folds 0-1; the best config is then
trained on folds 2-4 and reported with the stage report (models/model137/report.py).
  python run_model141.py search [n_configs]      -> runs/search.json
  python run_model141.py confirm                 -> best config on folds 2-4
  python run_model141.py seeds                   -> 5-seed bag of Model111's recipe, all folds
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402
import numpy as np  # noqa: E402

from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
BASE = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, max_depth=-1, min_child_samples=300,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.2, reg_lambda=3.0, max_bin=127)
SPACE = dict(num_leaves=[15, 31, 63, 127], learning_rate=[0.015, 0.03, 0.05], n_estimators=[600, 1000, 1600],
             min_child_samples=[100, 300, 1000], colsample_bytree=[0.3, 0.45, 0.6, 0.8], subsample=[0.6, 0.8, 1.0],
             reg_lambda=[0.0, 3.0, 10.0, 30.0], max_bin=[63, 127, 255])


class SeedShim:
    offset = 0
    import lightgbm as _lgb
    LGBMClassifier = _lgb.LGBMClassifier; Booster = _lgb.Booster

    @staticmethod
    def LGBMRegressor(**kw):
        import lightgbm as lgb
        return lgb.LGBMRegressor(**{**kw, "random_state": int(kw["random_state"]) + SeedShim.offset})


def setup():
    RUNS.mkdir(parents=True, exist_ok=True)
    for f in range(5):
        src = ROOT / f"models/model111/runs/fold{f}_teacher.npz"
        if src.exists() and not (RUNS / src.name).exists():
            shutil.copy2(src, RUNS / src.name)
    state = m111.load_everything()
    import os
    m111.RUNS = RUNS; m111.ALPHA = 0.5; m111.THREADS = int(os.environ.get("LGBM_THREADS", "8")); m111.lgb = SeedShim
    return state


def score(layout, pred, fold):
    held = layout.row_fold == fold
    ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
    return tsauc(layout.target[held], pred, layout.time_online[held]), tsauc(layout.target[held], ref, layout.time_online[held])


def run_config(state, tag, params, folds):
    _, protocol, layout, real, pool, teacher = state
    m111.m84.CONFIG["arms"]["expressive"] = {**BASE, **params}
    m111.TAG = tag
    out = {}
    for fold in folds:
        pred = m111.run_fold(protocol, layout, real, pool, teacher, fold)
        s, b = score(layout, pred, fold); out[fold] = s
        print(f"{tag} fold {fold}: {s:.6f} vs Model111 {b:.6f} ({s - b:+.6f}) params {params}", flush=True)
    return out


def main():
    cmd = sys.argv[1]
    state = setup()
    if cmd == "search":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        rng = np.random.default_rng(141)
        results = json.loads((RUNS / "search.json").read_text()) if (RUNS / "search.json").exists() else {}
        for i in range(n):
            params = {k: (v[rng.integers(len(v))].item() if hasattr(v[0], "item") else v[rng.integers(len(v))]) for k, v in SPACE.items()}
            tag = f"cfg{i:02d}"
            if tag in results:
                continue
            res = run_config(state, tag, params, [0, 1])
            results[tag] = {"params": params, "folds": res, "pilot_mean": float(np.mean(list(res.values())))}
            (RUNS / "search.json").write_text(json.dumps(results, indent=2))
        ref = 0.5 * (0.644007 + 0.607235)
        for tag, r in sorted(results.items(), key=lambda kv: -kv[1]["pilot_mean"]):
            print(f"{tag}: pilot {r['pilot_mean']:.5f} ({r['pilot_mean'] - ref:+.5f}) {r['params']}")
    elif cmd == "confirm":
        results = json.loads((RUNS / "search.json").read_text())
        best = max(results.items(), key=lambda kv: kv[1]["pilot_mean"])
        print("confirming", best[0], best[1]["params"], flush=True)
        run_config(state, best[0], best[1]["params"], [2, 3, 4])
    elif cmd == "seeds":
        for s in (1, 2, 3, 4):
            SeedShim.offset = 1000 * s
            run_config(state, f"seed{s}", {}, [0, 1, 2, 3, 4])
        SeedShim.offset = 0


if __name__ == "__main__":
    main()
