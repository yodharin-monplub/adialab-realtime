"""Absolute dev blend scores for several neural sets, on one common scale.

weights_eval172.py prints gains against a reference that itself contains the set being tested, so two
sets cannot be compared through it.  This prints the absolute five-fold blend mean per (ws, wg) and the
pilot-selected pair (folds 0-1 pick, folds 2-4 confirm), for every set named on the command line.
  python models/model172/compare_blends.py all synR all+new all+synR
A name may join several sets with '+': the logits are standardized and averaged, which is how a
DIVERSE-net ensemble is tested (Model168 only ever measured more nets of the SAME recipe).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

PATHS = {"all": "models/model170/runs/gpu/all_fold{k}_pred.npz",
         "new": "models/model169/runs/gpu/h256s0_fold{k}_pred.npz"}


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def gru_vector(name, L):
    """Logit vector for one net set, or the MEAN of several joined by '+' (a diverse-net ensemble)."""
    import numpy as np
    parts = name.split("+")
    out = np.zeros(L.row_count)
    for part in parts:
        pattern = PATHS.get(part, "models/model172/runs/gpu/" + part + "_fold{k}_pred.npz")
        v = np.empty(L.row_count)
        for k in range(5):
            v[L.row_fold == k] = lg(np.load(ROOT / pattern.format(k=k))["prediction"])
        out += (v - v.mean()) / v.std()          # standardize each set before averaging
    return out / len(parts)


def main(sets):
    L = load_development_layout(ROOT)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    cb = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    student = np.empty(L.row_count)
    for k in range(5):
        m = L.row_fold == k
        student[m] = (lg(np.load(ROOT / f"models/model164/runs/SRP40_fold{k}.npz")["prediction"]) - cb["mu_dist"]) / cb["sd_dist"]
    masks = [L.row_fold == k for k in range(5)]
    for name in sets:
        gru = gru_vector(name, L)
        gru = (gru - gru.mean()) / gru.std()
        rows = {}
        for ws in (0.5, 1.0, 1.5, 2.0):
            for wg in (0.25, 0.5, 0.75, 1.0, 1.5):
                v = chain + ws * student + wg * gru
                rows[(ws, wg)] = np.array([tsauc(L.target[m], v[m], L.time_online[m]) for m in masks])
        best = max(rows, key=lambda kv: rows[kv][:2].mean())          # pilot folds 0-1 choose
        f = rows[best]
        print(f"{name:8s} pilot-selected ws={best[0]} wg={best[1]}  folds {np.round(f, 5).tolist()}  "
              f"mean {f.mean():.5f}  (pilot {f[:2].mean():.5f} confirm {f[2:].mean():.5f})", flush=True)
        grid = {k: round(v.mean(), 5) for k, v in sorted(rows.items())}
        print(f"         absolute means by (ws,wg): {grid}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
