"""Model175: a DIFFERENT MODEL FAMILY as a fourth blend component.

Every component in the blend is a gradient-boosted tree or a GRU. What the blend has always paid for is
DIVERSITY - the GRU is worth +0.0016 in the blend while scoring only 0.62 alone. The chain does contain an
ExtraTrees and a RandomForest, but those were trained in August on development series only with HARD
labels, before the two levers that mattered were found (oracle-teacher soft targets, +0.007; all-series
training, +0.007). No other family has ever been given the current recipe.

This runs Model164's student assembly UNCHANGED - same rows, same 40% synthetic mass, same soft targets
0.5*y + 0.5*oracle-teacher, same columns - and swaps only the estimator, by replacing the LGBMRegressor
factory inside Model164's namespace with an adapter. So the comparison changes exactly one thing: the
learner.
  python models/model175/run_model175.py ridge 0          fast probe (minutes)
  python models/model175/run_model175.py extratrees 0 1   the real candidate (~30-40 min per fold)
Outputs models/model175/runs/{ARM}_fold{k}.npz with the same keys as the student's, so
models/model172/compare_blends.py and the weight protocol can read them.
A standalone TS-AUC far below the student's (~0.64 on a held fold) is fine - the question is only whether
it adds anything to the blend that the existing three parts do not already have.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import models.model164.run_model164 as m164  # noqa: E402

HERE = Path(__file__).resolve().parent


class Adapter:
    """Looks enough like LGBMRegressor for Model164's student() to drive it."""

    def __init__(self, kind, seed):
        self.kind = kind
        if kind == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler(copy=False)
            self.model = Ridge(alpha=10.0, random_state=seed)
        elif kind == "extratrees":
            from sklearn.ensemble import ExtraTreesRegressor
            self.scaler = None
            self.model = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=50, max_features=0.3,
                                             n_jobs=4, random_state=seed, bootstrap=False)
        else:
            raise SystemExit(f"unknown family {kind}")

    def fit(self, x, target, sample_weight=None):
        t0 = time.monotonic()
        if self.scaler is not None:
            x = self.scaler.fit_transform(np.nan_to_num(x, copy=False))
        self.model.fit(x, target, sample_weight=sample_weight)
        print(f"  {self.kind} fitted on {x.shape} in {time.monotonic() - t0:.0f}s", flush=True)
        return self

    def predict(self, x):
        if self.scaler is not None:
            x = self.scaler.transform(np.nan_to_num(x, copy=False))
        return self.model.predict(x)

    @property
    def booster_(self):
        adapter = self

        class _Saver:
            def save_model(self, path):                      # keep the fitted model for a later bundle
                import pickle
                Path(path).with_suffix(".pkl").write_bytes(pickle.dumps(adapter.model))
        return _Saver()


def main():
    kind = sys.argv[1]
    folds = [int(a) for a in sys.argv[2:]] or [0]
    arm = {"ridge": "RIDGE", "extratrees": "XTREES"}[kind]
    m164.RUNS = HERE / "runs"
    m164.RUNS.mkdir(parents=True, exist_ok=True)
    seen = {}

    def factory(**kwargs):                                   # Model164 passes LightGBM kwargs; ignore them
        return Adapter(kind, int(kwargs.get("random_state", 0)))

    m164.lgb.LGBMRegressor = factory
    print(f"{arm}: family {kind}, pooled sets {m164.TAGS}", flush=True)
    _, protocol, layout, real, pool, _ = m164.m147.m111.load_everything()
    rec = m164.load_pooled()
    from tsauc import tsauc
    for fold in folds:
        t0 = time.monotonic()
        m164.student(protocol, layout, real, pool, rec, fold, arm=arm)
        mine = m164.RUNS / f"{arm}_fold{fold}.npz"
        ref = HERE.parent / f"model164/runs/SRP40_fold{fold}.npz"
        if mine.exists() and ref.exists():
            m = layout.row_fold == fold
            a = tsauc(layout.target[m], np.load(mine)["prediction"], layout.time_online[m])
            b = tsauc(layout.target[m], np.load(ref)["prediction"], layout.time_online[m])
            seen[fold] = a
            print(f"{arm} fold {fold}: {a:.6f}  (the GBDT student is {b:.6f})  [{time.monotonic() - t0:.0f}s]",
                  flush=True)
    if seen:
        print(f"{arm} standalone mean over {sorted(seen)}: {np.mean(list(seen.values())):.6f}", flush=True)


if __name__ == "__main__":
    main()
