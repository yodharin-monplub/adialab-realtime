"""Select the Model112 top-level weight on pilot folds 0-1, confirm on folds 2-4.

chain = Submission28 OOF (Model26 stack with Model62 voters, all folds)
dist  = logit(Model111 distilled voter OOF)
cand  = z(chain) + w * z(dist), z using pooled development constants.
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
# Optional second argument: component file pattern with {k}; default = Model111 voter.
PATTERN = sys.argv[2] if len(sys.argv) > 2 else "models/model111/runs/{tag}_fold{k}.npz"
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "constants.json"
GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
PILOT, CONFIRM = (0, 1), (2, 3, 4)


def main():
    L = load_development_layout(ROOT)
    run62 = ROOT / json.loads((ROOT / "models/model62/runs/latest.json").read_text())["directory"]
    voter = np.empty(L.row_count, dtype=np.float32)
    dist = np.empty(L.row_count, dtype=np.float64)
    for k in range(5):
        m = L.row_fold == k
        voter[m] = np.load(run62 / f"lightgbm_aligned_40_fold{k}.npz")["prediction"]
        p = np.clip(np.load(ROOT / PATTERN.format(tag=TAG, k=k))["prediction"].astype(np.float64), 1e-6, 1 - 1e-6)
        dist[m] = np.log(p) - np.log1p(-p)
    chain = ensemble_margin(voter, L).astype(np.float64)
    np.save(HERE / "chain_m73_all.npy", chain.astype(np.float32))
    c = dict(mu_chain=float(chain.mean()), sd_chain=float(chain.std()),
             mu_dist=float(dist.mean()), sd_dist=float(dist.std()))
    zc = (chain - c["mu_chain"]) / c["sd_chain"]
    zd = (dist - c["mu_dist"]) / c["sd_dist"]

    def fold_score(v, k):
        m = L.row_fold == k
        return tsauc(L.target[m], v[m], L.time_online[m])

    base = np.array([fold_score(zc, k) for k in range(5)])
    table = {}
    for w in GRID:
        s = np.array([fold_score(zc + w * zd, k) for k in range(5)])
        table[w] = s
        print(f"w={w}: folds {np.round(s, 5).tolist()} mean {s.mean():.6f} gain {s.mean() - base.mean():+.6f}", flush=True)
    pilot_gain = {w: float((s - base)[list(PILOT)].mean()) for w, s in table.items()}
    weight = max(pilot_gain, key=pilot_gain.get)
    s = table[weight]
    gains = s - base
    summary = {
        "tag": TAG, "weight": weight, **c,
        "chain_fold_scores": base.tolist(), "chain_mean": float(base.mean()),
        "candidate_fold_scores": s.tolist(), "candidate_mean": float(s.mean()),
        "pilot_gain": float(gains[list(PILOT)].mean()),
        "confirmation_gain": float(gains[list(CONFIRM)].mean()),
        "confirmation_wins": int((gains[list(CONFIRM)] > 0).sum()),
        "pilot_gain_by_weight": pilot_gain,
    }
    summary["pattern"] = PATTERN
    (HERE / OUT_NAME).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
