"""Model112: evaluate deployable (Submission28-style) chains with distilled voters.

ensemble_margin(p) is exactly the Submission28 serving chain: Model26 stack
with the core voter column replaced by p, rolling-max-128 premium (0.1) and a
0.9/0.1 blend with the Model11 compact model.

Candidates per fold (only folds where Model111 predictions exist):
  m73      : voter = Model62 LightGBM aligned-40 (Submission28 as built)
  m84v     : voter = Model84 expressive LightGBM
  m111v    : voter = Model111 distilled expressive LightGBM
  X + w*z(logit m111) : top-level blend added to chain X
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for _name in ("fcntl", "resource"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.flock = lambda *a, **k: None
        _stub.LOCK_EX = _stub.LOCK_NB = 0
        _stub.getrusage = lambda *a, **k: types.SimpleNamespace(ru_maxrss=0)
        _stub.RUSAGE_SELF = 0
        sys.modules[_name] = _stub

import numpy as np  # noqa: E402

from models.model10.sources import load_development_layout  # noqa: E402
from models.model20.report import ensemble_margin  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "a050"


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def zs(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / x.std()


def voter(layout, loader, folds):
    old = np.load(ROOT / "models/model6/runs/tsroc0p64_909b8f5c6c/final_cv/core/xgboost__fisher_256__xgb_depth3/raw_oof.f32.npy")
    p = old.copy()
    for k in folds:
        p[layout.row_fold == k] = loader(k)
    return p


def main():
    L = load_development_layout(ROOT)
    run62 = ROOT / json.loads((ROOT / "models/model62/runs/latest.json").read_text())["directory"]
    run84 = ROOT / "models/model84/runs/e1e98a6e8af2df25"
    run111 = ROOT / "models/model111/runs"
    folds = [k for k in range(5) if (run111 / f"{TAG}_fold{k}.npz").exists()]
    load62 = lambda k: np.load(run62 / f"lightgbm_aligned_40_fold{k}.npz")["prediction"]
    load84 = lambda k: np.load(run84 / f"expressive_fold{k}.npz")["prediction"]
    load111 = lambda k: np.load(run111 / f"{TAG}_fold{k}.npz")["prediction"]
    chains = {name: ensemble_margin(voter(L, f, folds), L)
              for name, f in (("m73", load62), ("m84v", load84), ("m111v", load111))}
    np.save(HERE / f"chain_m73.npy", chains["m73"])
    rows = {}
    for k in folds:
        m = L.row_fold == k
        y, t = L.target[m], L.time_online[m]
        d = zs(lg(load111(k)))
        r = {}
        for name, c in chains.items():
            r[name] = tsauc(y, c[m], t)
            for w in (0.2, 0.3, 0.5):
                r[f"{name}+{w}"] = tsauc(y, zs(c[m]) + w * d, t)
        rows[k] = r
        print(k, {a: round(b, 5) for a, b in r.items()}, flush=True)
    keys = rows[folds[0]].keys()
    mean = {key: float(np.mean([rows[k][key] for k in folds])) for key in keys}
    print("mean", {a: round(b, 5) for a, b in mean.items()})
    (HERE / f"deployable_probe_{TAG}.json").write_text(json.dumps({"folds": rows, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
