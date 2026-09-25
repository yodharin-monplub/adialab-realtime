"""Is a FOURTH component worth a weight? blend = z(chain) + ws*z(student) + wg*z(nets) + w4*z(new).

The three-part reference is Submission42's: student weight 1.5, nets weight 1.0 (Model171). This searches
w4 on the pilot folds 0-1 and reports the confirmation folds 2-4 separately, the protocol every weight
decision in this project has used. A fourth component earns its place only if the pilot-selected w4 also
wins on the confirmation folds.
  python models/model175/blend4.py RIDGE [XTREES ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def available(pattern):
    """Folds whose prediction file exists - a component may still be training on the rest."""
    return [k for k in range(5) if (ROOT / pattern.format(k=k)).exists()]


def zfold(pattern, L, logit=True, folds=range(5)):
    v = np.zeros(L.row_count)
    for k in folds:
        p = np.load(ROOT / pattern.format(k=k))["prediction"]
        v[L.row_fold == k] = lg(p) if logit else np.asarray(p, np.float64)
    keep = np.isin(L.row_fold, list(folds))
    return (v - v[keep].mean()) / v[keep].std()


def main(names):
    L = load_development_layout(ROOT)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    cb = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    student = np.empty(L.row_count)
    for k in range(5):
        m = L.row_fold == k
        student[m] = (lg(np.load(ROOT / f"models/model164/runs/SRP40_fold{k}.npz")["prediction"]) - cb["mu_dist"]) / cb["sd_dist"]
    gru = zfold("models/model170/runs/gpu/all_fold{k}_pred.npz", L)
    masks = [L.row_fold == k for k in range(5)]

    def score(v):
        return np.array([tsauc(L.target[m], v[m], L.time_online[m]) for m in masks])

    base = score(chain + 1.5 * student + 1.0 * gru)
    print(f"Submission42 structure: folds {np.round(base, 5).tolist()} mean {base.mean():.5f}", flush=True)
    for name in names:
        # a linear model's output is already a logit-like score, not a probability: standardize as-is
        pattern = f"models/model175/runs/{name}_fold{{k}}.npz"
        folds = available(pattern)
        extra = zfold(pattern, L, logit=(name != "RIDGE"), folds=folds)
        if len(folds) < 5:
            print(f"    (only folds {folds} are trained so far - pilot reading only)", flush=True)
        print(f"--- {name}: correlation with the student {np.corrcoef(extra, student)[0, 1]:+.3f}, "
              f"with the nets {np.corrcoef(extra, gru)[0, 1]:+.3f}", flush=True)
        rows = {}
        for w4 in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            f = score(chain + 1.5 * student + 1.0 * gru + w4 * extra)
            rows[w4] = np.where(np.isin(np.arange(5), folds), f - base, np.nan)
            r = rows[w4]
            print(f"  w4={w4:<5} pilot {np.nanmean(r[:2]):+.5f}  confirm {np.nanmean(r[2:]) if not np.all(np.isnan(r[2:])) else float('nan'):+.5f}  "
                  f"all {np.nanmean(r):+.5f}  wins {int(np.nansum(r > 0))}/{int(np.sum(~np.isnan(r)))}", flush=True)
        best = max(rows, key=lambda w: np.nanmean(rows[w][:2]))
        d = rows[best]
        n = int(np.sum(~np.isnan(d)))
        if n < 5:
            print(f"  pilot-selected w4={best}: pilot {np.nanmean(d[:2]):+.5f} on folds {folds} "
                  f"- run the rest before judging", flush=True)
        else:
            verdict = "EARNS ITS PLACE" if np.nanmean(d[2:]) > 0 and int(np.nansum(d > 0)) >= 4 else "rejected"
            print(f"  pilot-selected w4={best}: confirm {np.nanmean(d[2:]):+.5f}, all {np.nanmean(d):+.5f}, "
                  f"wins {int(np.nansum(d > 0))}/5 -> {verdict}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["RIDGE"])
